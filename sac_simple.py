"""
标准SAC算法（离散动作）
- 双Q网络防止过估计
- 自动调节熵温度α
- 软更新目标网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """经验回放"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), np.array(action), 
                np.array(reward).reshape(-1, 1), 
                np.array(next_state), 
                np.array(done).reshape(-1, 1))
    
    def size(self):
        return len(self.buffer)


class Actor(nn.Module):
    """策略网络：输入状态，输出动作概率"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Q网络：输入状态+动作，输出Q值"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SAC:
    """标准SAC算法（离散动作）"""
    def __init__(self, state_dim, action_dim, config, device):
        self.device = device
        self.action_dim = action_dim  # 保存动作维度，避免硬编码
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.target_entropy = config['target_entropy']
        
        # 网络
        hidden_dim = config['hidden_dim']
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # 硬更新target
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=config['critic_lr']
        )
        
        # 可学习的温度参数α
        self.log_alpha = torch.tensor(np.log(config['alpha']), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['alpha_lr'])
    
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), probs.detach().cpu().numpy()
    
    def update(self, replay_buffer, batch_size):
        """更新网络参数 - 终极修复版（维度锁死）"""
        if replay_buffer.size() < batch_size:
            return
        
        # 1. 采样并转为 Tensor，强制 view 为 [batch_size, 1]
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        # 维度关键点：action, reward, done 必须全是 [batch_size, 1]
        action = torch.LongTensor(action).view(batch_size, 1).to(self.device)
        reward = torch.FloatTensor(reward).view(batch_size, 1).to(self.device)
        done = torch.FloatTensor(done).view(batch_size, 1).to(self.device)
        
        # 预准备：所有动作的 one-hot
        all_actions = torch.arange(self.action_dim, device=self.device)
        all_actions_onehot = F.one_hot(all_actions, num_classes=self.action_dim).float() 

        # ========== 1. 计算 Target Q 值 (Expectation SAC) ==========
        with torch.no_grad():
            next_probs = self.actor(next_state) # [batch, action_dim]
            next_log_probs = torch.log(next_probs + 1e-8)
            
            # 扩展状态和动作以匹配: [batch, action_dim, dim]
            next_state_exp = next_state.unsqueeze(1).repeat(1, self.action_dim, 1)
            all_act_exp = all_actions_onehot.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 获取所有动作的 Q 值并去掉最后一维 -> [batch, action_dim]
            t_q1_all = self.target_critic1(next_state_exp, all_act_exp).view(batch_size, self.action_dim)
            t_q2_all = self.target_critic2(next_state_exp, all_act_exp).view(batch_size, self.action_dim)
            t_q_min_all = torch.min(t_q1_all, t_q2_all)
            
            # 计算 V(s') = sum( pi * (Q - alpha * log_pi) )
            alpha = self.log_alpha.exp()
            target_v = (next_probs * (t_q_min_all - alpha * next_log_probs)).sum(dim=1, keepdim=True) # [batch, 1]
            
            # 最终 Target Q: [batch, 1]
            target_q = reward + self.gamma * (1 - done) * target_v

        # ========== 2. 更新 Critic ==========
        action_onehot = F.one_hot(action.squeeze(1), num_classes=self.action_dim).float()
        current_q1 = self.critic1(state, action_onehot).view(batch_size, 1)
        current_q2 = self.critic2(state, action_onehot).view(batch_size, 1)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== 3. 更新 Actor (全概率更新) ==========
        probs = self.actor(state) # [batch, action_dim]
        log_probs = torch.log(probs + 1e-8)
        
        with torch.no_grad():
            state_exp = state.unsqueeze(1).repeat(1, self.action_dim, 1)
            q1_all = self.critic1(state_exp, all_act_exp).view(batch_size, self.action_dim)
            q2_all = self.critic2(state_exp, all_act_exp).view(batch_size, self.action_dim)
            q_min_all = torch.min(q1_all, q2_all)
        
        # Actor Loss: 离散形式不需要采样，直接对所有动作加权平均
        # 目标是最小化: E[alpha * log_pi - Q]，即最大化熵和Q
        inside_term = self.log_alpha.exp().detach() * log_probs - q_min_all
        actor_loss = (probs * inside_term).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== 4. 更新温度参数 alpha ==========
        with torch.no_grad():
            # 计算当前分布的平均负熵
            current_entropy = -(probs * log_probs).sum(dim=1).mean() 
        
        # 离散 SAC 的 alpha 更新：alpha * (H_current - H_target)
        # 注意：target_entropy 应该是一个正数
        alpha_loss = (self.log_alpha.exp() * (current_entropy.detach() - self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ========== 5. 软更新目标网络 ==========
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)