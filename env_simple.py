"""
极简版边缘环境
- 2个节点，1个用户
- 状态：[任务大小(Mbits), 节点0积压(Gcycles), 节点1积压(Gcycles)]
- 动作：0 或 1（选哪个节点）
- 奖励：-(计算时延 + 排队时延)
"""
import numpy as np

class SimpleEdgeEnv:
    def __init__(self, config):
        # ========== 基础参数 ==========
        self.num_nodes = config['num_nodes']
        self.num_users = config.get('num_users', 1)
        self.time_slots = config['slots_per_episode']
        
        # ========== 状态和动作维度 ==========
        self.state_dim = 1 + self.num_nodes
        self.action_dim = self.num_nodes
        
        # ========== 配置验证 ==========
        required_keys = ['node_capacities', 'data_range', 'compute_density', 'delta']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # ========== 输入参数（人类可读）==========
        node_capacities_ghz = config['node_capacities']  # [10, 20] GHz
        compute_density = config['compute_density']      # 200 cycles/bit
        data_range_mbits = config['data_range']          # [10, 40] Mbits
        self.delta = config['delta']                     # 1.0 秒
        
        # ========== 内部单位统一 ==========
        # 节点算力：GHz → cycles/s
        self.node_capacities = np.array(node_capacities_ghz) * 1e9
        # 数据量：Mbits → bits
        self.data_range = np.array(data_range_mbits) * 1e6
        # 计算密度：cycles/bit（不变）
        self.compute_density = compute_density
        
        # ========== 动态变量 ==========
        self.backlog = np.zeros(self.num_nodes)  # cycles
        self.current_slot = 0
        self.current_data = 0.0                  # bits
        self.current_compute = 0.0               # cycles
        
        # ========== 随机种子 ==========
        if 'seed' in config:
            np.random.seed(config['seed'])
    
    def reset(self):
        """重置环境，返回初始状态（归一化后）"""
        self.backlog = np.zeros(self.num_nodes)
        self.current_slot = 0
        
        # 生成第一个任务（单位：bits）
        self.current_data = np.random.uniform(self.data_range[0], self.data_range[1])
        self.current_compute = self.current_data * self.compute_density
        
        return self._get_state()
    
    def _get_state(self):
        """
        获取归一化后的状态
        - 任务大小：bits → Mbits (范围 10-40)
        - 节点积压：cycles → Gcycles (动态节点数)
        """
        task_size_mbits = np.array([self.current_data / 1e6], dtype=np.float32)
        backlog_gcycles = (self.backlog/1e9).astype(np.float32)
        return np.concatenate([task_size_mbits, backlog_gcycles], axis=0)
    
    def step(self, action):
        """
        执行一步
        action: 0 到 num_nodes-1，选择节点
        """
        # 动态验证动作范围
        if not (0 <= action < self.num_nodes):
            raise ValueError(f"动作必须在 [0, {self.num_nodes-1}] 范围内，收到: {action}")
        
        node = action
        '''
        print(f"\n[DEBUG Step {self.current_slot}] Node Action: {node}")
        print(f"  Current Data: {self.current_data/1e6:.2f} Mbits")
        print(f"  Current Compute Needed: {self.current_compute/1e9:.4f} Gcycles")
        print(f"  Backlog Before: {self.backlog/1e9} Gcycles")
        # 1. 计算时延（内部单位已统一）
        '''
        comp_delay = self.current_compute / self.node_capacities[node]
        queue_delay = self.backlog[node] / self.node_capacities[node]
        total_delay = comp_delay + queue_delay
        reward=-total_delay
        
        # 2. 更新积压（任务加入队列）
        self.backlog[node] += self.current_compute
        
        # 3. 时隙结束，节点处理任务
        self._update_queues()
        
        # 4. 生成下一个任务
        self.current_data = np.random.uniform(self.data_range[0], self.data_range[1])
        self.current_compute = self.current_data * self.compute_density
        next_state = self._get_state()
        
        # 5. 时隙计数
        self.current_slot += 1
        done = bool(self.current_slot >= self.time_slots)
        
        info = {
            'node': node,
            'comp_delay': comp_delay,
            'queue_delay': queue_delay,
            'total_delay': total_delay,
            'reward': reward
        }
        return next_state, reward, done, info
    
    def _update_queues(self):
        """时隙结束时，更新节点积压"""
        processed = self.node_capacities * self.delta  # cycles
        '''
        print(f"  [QUEUE CHECK] Capacity: {self.node_capacities/1e9} GHz")
        print(f"  [QUEUE CHECK] Delta: {self.delta} s")
        print(f"  [QUEUE CHECK] Processed in this step: {processed/1e9} Gcycles")
        '''
        self.backlog = np.maximum(self.backlog - processed, 0)