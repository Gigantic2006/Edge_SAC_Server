import numpy as np

class ComputeNode:
    """边缘节点类：负责物理层积压消耗"""
    def __init__(self, node_id, capacity_ghz, pos=(0,0)):
        self.id = node_id
        self.capacity = capacity_ghz * 1e9  # 转化为 Hz
        self.pos = np.array(pos)
        self.backlog = 0.0                  # 单位: cycles

    def process(self, delta):
        """物理时钟推进：在一个大时隙结束时调用"""
        processed = self.capacity * delta
        self.backlog = max(0, self.backlog - processed)

class AdvancedEdgeEnv:
    """
    高级边缘计算环境：
    支持多用户逐一决策（Micro-step），为数字孪生和实时风险控制预留接口。
    """
    def __init__(self, config):
        # 1. 初始化节点列表
        num_nodes = len(config['node_capacities'])
        node_positions = config.get('node_positions', [(0, 0)] * num_nodes)
        self.nodes = [
            ComputeNode(i, cap, pos=node_positions[i] if i < len(node_positions) else (0, 0)) 
            for i, cap in enumerate(config['node_capacities'])
        ]
        self.num_nodes = len(self.nodes)
        
        # 2. 配置参数
        self.num_users = config.get('num_users', 3)   # 每个时隙的用户数
        self.data_range = np.array(config['data_range']) * 1e6 # MB -> Bits
        self.comp_density = config['compute_density'] # cycles per bit
        self.delta = config['delta']                   # 时隙长度 (s)
        self.slots_per_episode = config['slots_per_episode']
        
        # 3. 算法接口维度 (固定维度，不随用户数改变)
        self.state_dim = 2 + self.num_nodes  # [Dn, Cn] + [M个节点的Backlog]
        self.action_dim = self.num_nodes     # 动作：选择 0 ~ M-1 号节点
        
        self.reset()

    def reset(self):
        """重置环境：大时隙、用户指针、节点状态全部归零"""
        for node in self.nodes:
            node.backlog = 0.0
        self.current_slot = 0
        self.user_ptr = 0  # 指向当前待处理的用户 (0 到 num_users-1)
        self._generate_multi_tasks()
        return self._get_obs()

    def _generate_multi_tasks(self):
        """每个大时隙开始时，预先生成本时隙所有用户的任务元组"""
        sizes = np.random.uniform(self.data_range[0], self.data_range[1], self.num_users)
        self.current_tasks = [
            {'bits': s, 'cycles': s * self.comp_density} for s in sizes
        ]

    def _get_obs(self):
        """获取当前待决策用户的观测状态（实现数字孪生的实时状态反馈）"""
        task = self.current_tasks[self.user_ptr]
        obs = [
            task['bits'] / 1e6,    # 当前任务数据量 (归一化到 Mbits)
            task['cycles'] / 1e9   # 当前任务计算量 (归一化到 Gcycles)
        ]
        # 拼接所有节点的实时积压情况
        for node in self.nodes:
            obs.append(node.backlog / 1e9) # 归一化到 Gcycles
            
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        执行微步决策：
        1. 接收当前用户的动作。
        2. 更新节点积压（立即生效，影响下一微步的 obs）。
        3. 如果时隙结束，推进物理时钟。
        """
        target_node = self.nodes[action]
        task = self.current_tasks[self.user_ptr]
        
        # A. 计算基础指标 (计算延迟 = 排队延迟 + 处理延迟)
        wait_time = target_node.backlog / target_node.capacity
        proc_time = task['cycles'] / target_node.capacity
        delay = wait_time + proc_time
        
        # B. 数字孪生/实时更新：决策后的任务立即进入队列积压
        target_node.backlog += task['cycles']
        
        # 粗略估计最大延迟：(数据量范围上界) / (最小节点容量) + backlog / capacity
        # 不要复杂的归一化了，直接给一个适中的惩罚量级
        reward = -delay * 0.1  # 缩放因子
        reward = max(reward, -2.0) # 截断，防止初始 backlog 爆炸导致梯度崩溃
        
        # D. 推进指针与逻辑判断
        self.user_ptr += 1
        done = False
        
        # 检查一个大时隙是否结束
        if self.user_ptr >= self.num_users:
            self.user_ptr = 0 # 重置用户指针
            # 物理演进：所有节点在这个时隙内并行处理任务
            for node in self.nodes:
                node.process(self.delta)
            
            self.current_slot += 1
            # 检查整个 Episode 是否结束
            if self.current_slot >= self.slots_per_episode:
                done = True
            else:
                self._generate_multi_tasks() # 为下一时隙生成新任务集

        return self._get_obs(), reward, done, {'delay': delay}