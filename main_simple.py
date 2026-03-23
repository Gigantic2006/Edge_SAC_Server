import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from env_simple import AdvancedEdgeEnv 
from sac_simple import SAC, ReplayBuffer

def main():
    # ========== 1. 实验配置 ==========
    config = {
        "exp_name": "SAC_Edge_MicroStep_v1",
        "seed": 42,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
        # 环境参数
        "env": {
            "num_users": 3,              # 每个时隙的用户数
            "node_capacities": [14, 16],  # 两个边缘节点的算力 (GHz)
            "compute_density": 300,      # 计算强度 (cycles/bit)
            "data_range": [10, 50],      # 任务大小范围 (Mbits)
            "slots_per_episode": 100,    # 每个 Episode 包含的时隙数
            "delta": 1.0,                # 时隙物理时长 (s)
        },
        
        # 算法参数
        "sac": {
            "hidden_dim": 256,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 1e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_capacity": 100000,
            "batch_size": 256,
            "update_freq": 2,            # 每个 micro-step 更新多少次模型
            "warmup_steps": 500,         # 初始随机采样步数
        },
        
        "train": {
            "total_episodes": 50,       # 训练总轮数
            "save_interval": 20,
        }
    }

    # ========== 2. 初始化 ==========
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    env = AdvancedEdgeEnv(config["env"])
    
    # 创建模型保存目录
    os.makedirs(f"runs/{config['exp_name']}", exist_ok=True)
    
    # 动态计算离散 SAC 的目标熵: 0.98 * log(|A|)
    config["sac"]["target_entropy"] = 0.4
    
    agent = SAC(env.state_dim, env.action_dim, config["sac"], config["device"])
    replay_buffer = ReplayBuffer(config["sac"]["buffer_capacity"])
    writer = SummaryWriter(log_dir=f"runs/{config['exp_name']}")
    
    # ========== 3. 训练主循环 ==========
    global_step = 0
    best_reward = -float('inf')
    history_rewards = []
    history_delays = []

    for epi in range(config["train"]["total_episodes"]):
        state = env.reset()
        epi_reward = 0
        epi_delays = []
        
        # 【关键改动】: 使用 while True 自动处理 (用户数 * 时隙数) 的步数
        while True:
            # 预热期判定
            if global_step < config["sac"]["warmup_steps"]:
                action = np.random.randint(0, env.action_dim)
            else:
                action, _ = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            # 更新模型（每隔 update_freq 步才更新一次）
            if (
                global_step >= config["sac"]["warmup_steps"]
                and replay_buffer.size() >= config["sac"]["batch_size"]
                and global_step % config["sac"]["update_freq"] == 0
            ):
                losses = agent.update(replay_buffer, config["sac"]["batch_size"])
                
                if global_step % 100 == 0:
                    for k, v in losses.items():
                        writer.add_scalar(f"Loss/{k}", v, global_step)

            state = next_state
            epi_reward += reward
            epi_delays.append(info['delay']) # 记录每个用户的延迟
            global_step += 1
            
            if done: # 当 slots_per_episode 消耗完且最后一个用户处理完时，env 会返回 done
                break

        # ========== 4. 日志记录 ==========
        avg_delay = np.mean(epi_delays)
        history_rewards.append(epi_reward)
        history_delays.append(avg_delay)
        
        writer.add_scalar("Rollout/Episode_Reward", epi_reward, epi)
        writer.add_scalar("Rollout/Avg_Delay", avg_delay, epi)
        writer.add_scalar("Rollout/Alpha", agent.alpha(), epi)

        print(f"Epi: {epi+1:03d} | Reward: {epi_reward:.2f} | Avg_Delay: {avg_delay:.4f}s | Alpha: {agent.alpha():.4f}")

        # 保存最优模型
        if epi_reward > best_reward:
            best_reward = epi_reward
            torch.save(agent.actor.state_dict(), f"runs/{config['exp_name']}/actor_best.pth")

    writer.close()
    
    # ========== 5. 绘制收敛曲线 ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(history_delays, color='red')
    ax1.set_ylabel('Avg Delay (s)')
    ax1.set_title('Training Convergence')
    ax1.grid(True)

    ax2.plot(history_rewards, color='blue')
    ax2.set_ylabel('Total Reward')
    ax2.set_xlabel('Episode')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=300)
    print("\n训练完成！收敛图已保存至 convergence.png")

if __name__ == "__main__":
    main()