import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from collections import deque
import numpy as np
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1) # 离散动作输出概率分布
        )
    
    def forward(self, state):
        return self.net(state)

class DiscreteCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # 输出 Q(s, a1), Q(s, a2)...
        )

    def forward(self, state):
        return self.net(state)

class ReplayBuffer:
    """经验回放池：负责存储和随机采样"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 解包并快速转换为 NumPy 数组，提高进入 Tensor 之前的处理速度
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), 
                np.array(action), 
                np.array(reward).reshape(-1, 1), 
                np.array(next_state), 
                np.array(done).reshape(-1, 1))
    
    def size(self):
        return len(self.buffer)
class SAC:
    def __init__(self, state_dim, action_dim, config, device):
        self.device = device
        self.action_dim = action_dim
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.target_entropy = config['target_entropy']
        
        # 网络初始化
        self.actor = Actor(state_dim, action_dim, config['hidden_dim']).to(device)
        self.critic1 = DiscreteCritic(state_dim, action_dim, config['hidden_dim']).to(device)
        self.critic2 = DiscreteCritic(state_dim, action_dim, config['hidden_dim']).to(device)
        self.target_critic1 = DiscreteCritic(state_dim, action_dim, config['hidden_dim']).to(device)
        self.target_critic2 = DiscreteCritic(state_dim, action_dim, config['hidden_dim']).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=config['critic_lr']
        )
        
        # 自动熵增益 alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['alpha_lr'])

    def alpha(self):
        # 使用 .item() 将 Tensor 转换为 Python 的 float 数值
        return self.log_alpha.exp().item()

    def select_action(self, state):
        """选择动作：根据当前策略输出概率并采样"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
        
        # 使用 Categorical 分布进行采样（保持探索性）
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), probs.detach().cpu().numpy()
    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # 转为 Tensor (保持 batch_size, dim 形式)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).view(-1, 1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # ------------------ 1. 更新 Critic ------------------
        with torch.no_grad():
            next_probs = self.actor(next_state)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1 = self.target_critic1(next_state)
            next_q2 = self.target_critic2(next_state)
            next_q = torch.min(next_q1, next_q2)
            alpha = self.log_alpha.exp()
            
            # 离散 SAC 的期望计算: V(s') = \sum \pi * (Q - \alpha * log \pi)
            target_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward + self.gamma * (1 - done) * target_v

        # 这里的 gather 代替了复杂的 one_hot 矩阵乘法
        current_q1 = self.critic1(state).gather(1, action)
        current_q2 = self.critic2(state).gather(1, action)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------ 2. 更新 Actor ------------------
        probs = self.actor(state)
        log_probs = torch.log(probs + 1e-8)
        
        with torch.no_grad():
            q1 = self.critic1(state)
            q2 = self.critic2(state)
            min_q = torch.min(q1, q2)
            alpha_detached = self.log_alpha.exp().detach()
        
        # Actor Loss: 最小化 KL 散度，即最小化 (\alpha * log \pi - Q) 的期望
        actor_loss = (probs * (alpha_detached * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------ 3. 更新 Alpha ------------------
        entropy = -(probs * log_probs).sum(dim=1).detach()
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------ 4. 软更新 ------------------
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)
        
        return {"a_loss": actor_loss.item(), "c_loss": critic_loss.item(), "alpha": self.log_alpha.exp().item()}

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)