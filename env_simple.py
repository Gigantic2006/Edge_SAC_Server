import numpy as np

class ComputeNode:
    """节点类：封装节点属性和内部逻辑"""
    def __init__(self, node_id, capacity_ghz, pos=(0,0)):
        self.id = node_id
        self.capacity = capacity_ghz * 1e9  # Hz
        self.pos = np.array(pos)            # 预留位置信息，方便以后加通信成本
        self.backlog = 0.0                  # cycles
        self.energy_consumed = 0.0          # 预留能耗统计

    def process(self, delta):
        """时隙结束时的处理逻辑"""
        processed = self.capacity * delta
        self.backlog = max(0, self.backlog - processed)

class SimpleEdgeEnv:
    def __init__(self, config):
        # 1. 动态节点生成 (扩展性：改config即可增加节点)
        self.nodes = [
            ComputeNode(i, cap) 
            for i, cap in enumerate(config['node_capacities'])
        ]
        self.num_nodes = len(self.nodes)
        
        # 2. 任务参数
        self.compute_density = config['compute_density']
        self.data_range = np.array(config['data_range']) * 1e6
        self.delta = config['delta']
        self.slots_per_episode = config['slots_per_episode']
        
        # 3. 自动计算维度 (扩展性：无论加多少节点，算法层自动适配)
        self.state_dim = 1 + self.num_nodes 
        self.action_dim = self.num_nodes
        
        self.reset()

    def reset(self):
        """重置环境"""
        for node in self.nodes:
            node.backlog = 0.0
        self.current_slot = 0
        self._generate_new_task()
        return self._get_obs()

    def _generate_new_task(self):
        """任务生成逻辑 (扩展性：以后可以改为泊松分布或其他模型)"""
        size_bits = np.random.uniform(self.data_range[0], self.data_range[1])
        self.current_task = {
            'bits': size_bits,
            'cycles': size_bits * self.compute_density
        }

    def _get_obs(self):
        """
        获取状态 (扩展性：以后增加输入维度只需在此 append)
        注意：一定要保持顺序一致
        """
        obs = []
        # 维度 1: 当前任务大小 (归一化到 Mbits)
        obs.append(self.current_task['bits'] / 1e6)
        # 维度 2~N: 各节点积压 (归一化到 Gcycles)
        for node in self.nodes:
            obs.append(node.backlog / 1e9)
        
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, info):
        """
        奖励计算中心 (扩展性：以后增加通信、迁移成本只需在此加减项)
        """
        # 目前只包含：-(计算延迟 + 排队延迟)
        reward = -(info['comp_delay'] + info['queue_delay'])
        return reward

    def step(self, action):
        """执行动作"""
        target_node = self.nodes[action]
        
        # 1. 计算当前任务的时延指标
        comp_delay = self.current_task['cycles'] / target_node.capacity
        queue_delay = target_node.backlog / target_node.capacity
        total_delay = comp_delay + queue_delay
        
        # 2. 封装信息 (扩展性：为了以后计算复杂的 Reward)
        step_info = {
            'comp_delay': comp_delay,
            'queue_delay': queue_delay,
            'total_delay': total_delay,
            'task_size': self.current_task['bits']
        }
        
        # 3. 计算 Reward
        reward = self._compute_reward(step_info)
        
        # 4. 更新物理状态
        target_node.backlog += self.current_task['cycles'] # 任务入队
        for node in self.nodes:
            node.process(self.delta) # 时隙内节点处理任务
            
        # 5. 推进时隙
        self.current_slot += 1
        self._generate_new_task() # 生成下一时隙的任务
        
        done = self.current_slot >= self.slots_per_episode
        next_state = self._get_obs()
        
        return next_state, reward, done, step_info