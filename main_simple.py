import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from env_simple import SimpleEdgeEnv
from sac_simple import SAC, ReplayBuffer

def main():
    # ========== 1. 实验配置 (未来可抽离到 yaml) ==========
    config = {
        "exp_name": "SAC_Edge_Baseline_v1",
        "seed": 42,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
        # 环境参数
        "env": {
            "num_nodes": 2,
            "node_capacities": [5, 10],  # GHz
            "compute_density": 300,      # cycles/bit
            "data_range": [20, 60],      # Mbits
            "slots_per_episode": 200,
            "delta": 1.0,
        },
        
        # 算法参数
        "sac": {
            "hidden_dim": 256,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "target_entropy": 0.4,       # 离散 SAC 通常设为 0.6 * log(action_dim)
            "buffer_capacity": 100000,
            "batch_size": 256,
            "update_freq": 2,            # 每一轮 step 更新几次
            "warmup_steps": 1000,        # 预热期：先随机采样，不训练
        },
        
        # 训练过程
        "train": {
            "total_episodes": 50,
            "save_interval": 50,
        }
    }

    # ========== 2. 初始化模块 ==========
    # 设置随机种子
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    env = SimpleEdgeEnv(config["env"])
    agent = SAC(env.state_dim, env.action_dim, config["sac"], config["device"])
    replay_buffer = ReplayBuffer(config["sac"]["buffer_capacity"])
    
    # 日志记录器 (TensorBoard)
    writer = SummaryWriter(log_dir=f"runs/{config['exp_name']}")
    
    # ========== 3. 训练主循环 ==========
    global_step = 0
    best_reward = -float('inf')
    
    history_rewards = []
    history_delays = []
    for epi in range(config["train"]["total_episodes"]):
        state = env.reset()
        epi_reward = 0
        epi_delay = []

        for step in range(config["env"]["slots_per_episode"]):
            # 预热期判定
            if global_step < config["sac"]["warmup_steps"]:
                action = np.random.randint(0, env.action_dim)
                probs = np.ones(env.action_dim) / env.action_dim
            else:
                action, probs = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            # 更新模型
            if (
                global_step >= config["sac"]["warmup_steps"]
                and replay_buffer.size() >= config["sac"]["batch_size"]
                and global_step % config["sac"]["update_freq"] == 0
            ):
                losses = agent.update(replay_buffer, config["sac"]["batch_size"])
                if global_step % 10 == 0: # 减少记录频率，节省 IO
                    for k, v in losses.items():
                        writer.add_scalar(f"Loss/{k}", v, global_step)

            state = next_state
            epi_reward += reward
            epi_delay.append(info['total_delay'])
            global_step += 1
            
            if done: break

        # ========== 4. 周期性记录与打印 ==========
        avg_delay = np.mean(epi_delay)
        history_rewards.append(epi_reward)
        history_delays.append(avg_delay)
        writer.add_scalar("Rollout/Episode_Reward", epi_reward, epi)
        writer.add_scalar("Rollout/Avg_Delay", avg_delay, epi)

        
        print(f"Epi: {epi+1:03d} | Reward: {epi_reward:.2f} | Delay: {avg_delay:.2f}s | Alpha: {agent.alpha():.4f}")

        # 保存最优模型
        if epi_reward > best_reward:
            best_reward = epi_reward
            torch.save(agent.actor.state_dict(), f"runs/{config['exp_name']}/actor_best.pth")

        # 周期保存，避免长训练中断导致结果丢失
        if (epi + 1) % config["train"]["save_interval"] == 0:
            torch.save(agent.actor.state_dict(), f"runs/{config['exp_name']}/actor_ep{epi+1}.pth")

    writer.close()
    print("\n训练结束，模型已保存至 runs/ 文件夹。")
    # ========== 5. 训练结束，直接画图 ==========
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 画延迟曲线
    ax1.plot(history_delays, color='red', label='Avg Delay')
    ax1.set_ylabel('Delay (s)')
    ax1.set_title('Training Progress')
    ax1.grid(True)
    
    # 画奖励曲线
    ax2.plot(history_rewards, color='blue', label='Episode Reward')
    ax2.set_ylabel('Reward')
    ax2.set_xlabel('Episode')
    ax2.grid(True)
    plt.tight_layout()
    save_path = "convergence.png"
    plt.savefig(save_path, dpi=300) # dpi=300 保证图片是高清的
    print(f"\n成功！收敛曲线图已保存至: {save_path}")
    
if __name__ == "__main__":
    main()