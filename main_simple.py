"""
极简版训练主程序
"""
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from env_simple import SimpleEdgeEnv
from sac_simple import SAC, ReplayBuffer

def set_seed(seed):
    """统一设置随机种子，确保可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # ========== 1. 环境参数 ==========
    env_config = {
        'num_nodes': 2,
        'node_capacities': [5, 10],  # GHz
        'compute_density': 300,        # cycles/bit
        'data_range': [20, 60],        # Mbits
        'slots_per_episode': 200,
        'delta': 1,                  # 秒
        'seed': 42                     # 环境种子
    }
    
        # ========== 2. 模型参数 ==========
    sac_config = {
        'state_dim': None,
        'action_dim': None,
        'hidden_dim': 128,      # 简单环境 128 够了
        'actor_lr': 1e-4,       # 从 1e-6 调大！1e-6 慢到几乎没更新，建议 3e-4 或 1e-4
        'critic_lr': 1e-3,      # 调大！Critic 需要学得比 Actor 快，建议 1e-3
        'alpha_lr': 3e-4,
        'gamma': 0.95,          # 没问题
        'tau': 0.005,           # 没问题
        'alpha': 1.0,           # 初始值设为 1.0 比较稳妥
        'target_entropy': 0.4,  # 重要修正！离散动作：0.6 * log(2) ≈ 0.4
        'buffer_capacity': 10000, # 建议稍微加大，防止旧数据太快被刷掉
        'batch_size': 256,
        'update_freq': 2,       # 修正！建议设为 2，即每步都更新。你总共才 300 轮，更新频率低了学不出来
    }
    
    # ========== 3. 训练参数 ==========
    train_config = {
        'episodes': 100,
        'seed': 42,                    # 全局种子
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    # ========== 4. 设置随机种子 ==========
    set_seed(train_config['seed'])
    env_config['seed'] = train_config['seed']  # 环境也使用相同种子
    
    print(f"使用设备: {train_config['device']}")
    print(f"随机种子: {train_config['seed']}")
    
    # ========== 5. 初始化 ==========
    env = SimpleEdgeEnv(env_config)
    sac_config['state_dim'] = env.state_dim
    sac_config['action_dim'] = env.action_dim

    agent = SAC(sac_config['state_dim'], 
                sac_config['action_dim'], 
                sac_config, 
                train_config['device'])
    replay_buffer = ReplayBuffer(sac_config['buffer_capacity'])
    
    # ========== 6. 训练循环 ==========
    episode_costs = []
    global_step = 0  # 全局步数计数器，不随 episode 重置
    
    for episode in range(train_config['episodes']):
        state = env.reset()
        episode_cost = 0
        episode_physical_delay = 0
        
        for step in range(env_config['slots_per_episode']):
            # 选择动作
            action, probs = agent.select_action(state)
            if step % 50 == 0 and step >0:
              # 格式化打印：[任务大小, 节点0积压, 节点1积压] -> [选0概率, 选1概率]
              print(f"Epi {episode} | State: {state} | Probs: {probs[0]} {episode}")
            next_state, reward, done, info = env.step(action)
            episode_cost += reward
            episode_physical_delay += info['total_delay']
            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新网络（使用全局步数，每 update_freq 步更新一次）
            if global_step % sac_config['update_freq'] == 0 and replay_buffer.size() >= sac_config['batch_size']:
                agent.update(replay_buffer, sac_config['batch_size'])
            
            state = next_state
            global_step += 1
            
            if done:
                break
        
        # 记录总时延（正数，方便看）
        avg_episode_delay = episode_physical_delay / env_config['slots_per_episode']
        episode_costs.append(avg_episode_delay)
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_cost = np.mean(episode_costs[-50:])
            print(f"Episode {episode+1}, Avg Delay: {avg_cost:.2f}s, Buffer Size: {replay_buffer.size()}")
    
    # ========== 7. 保存结果和画图 ==========
    #np.savetxt('episode_costs.csv', episode_costs, delimiter=',')
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_costs)
    plt.xlabel('Episode')
    plt.ylabel('Total Delay (s)')
    plt.title('Training Convergence')
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence.png', dpi=300)
    #plt.savefig('convergence.pdf', bbox_inches='tight')
    plt.show()
    
    print("=" * 50)
    print("训练完成！结果已保存。")
    print(f"最终平均时延 (最后100轮): {np.mean(episode_costs[-100:]):.2f}s")
    print(f"最优时延: {np.min(episode_costs):.2f}s")
    print("=" * 50)

if __name__ == '__main__':
    main()