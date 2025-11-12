import torch.nn as nn
import math
from typing import List, Dict, Any

class LayerSampler:
    """层采样器 - 从基础模型中均匀采样层"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.draft_config = config['draft_model']
        
    def get_uniform_indices(self, total_layers: int) -> List[int]:
        """获取均匀采样索引"""
        num_sampled = self.draft_config['num_sampled_layers']
        
        # 如果配置中指定了具体索引，直接使用
        if 'sampled_indices' in self.draft_config and self.draft_config['sampled_indices']:
            indices = self.draft_config['sampled_indices']
            if len(indices) == num_sampled:
                return [min(idx, total_layers - 1) for idx in indices]
        
        # 自动计算均匀采样索引
        strategy = self.draft_config.get('sampling_strategy', 'uniform')
        
        if strategy == "uniform":
            return self._uniform_sampling(total_layers, num_sampled)
        elif strategy == "geometric":
            return self._geometric_sampling(total_layers, num_sampled)
        elif strategy == "logarithmic":
            return self._logarithmic_sampling(total_layers, num_sampled)
        else:
            return self._uniform_sampling(total_layers, num_sampled)
    
    def _uniform_sampling(self, total_layers: int, num_sampled: int) -> List[int]:
        """均匀采样"""
        if num_sampled >= total_layers:
            return list(range(total_layers))
        
        step = total_layers / num_sampled
        indices = [int(i * step) for i in range(num_sampled)]
        
        # 确保包含首层和尾层
        indices[0] = 0
        indices[-1] = total_layers - 1
        
        # 确保索引不重复且有序
        indices = sorted(set(indices))
        while len(indices) < num_sampled:
            # 补充中间层
            for i in range(len(indices) - 1):
                if len(indices) >= num_sampled:
                    break
                mid = (indices[i] + indices[i + 1]) // 2
                if mid not in indices:
                    indices.append(mid)
            indices.sort()
        
        return indices[:num_sampled]
    
    def _geometric_sampling(self, total_layers: int, num_sampled: int) -> List[int]:
        """几何采样 - 更多关注深层"""
        if num_sampled >= total_layers:
            return list(range(total_layers))
        
        indices = []
        for i in range(num_sampled):
            # 使用几何分布，更多采样深层
            ratio = (i / (num_sampled - 1)) ** 1.5
            idx = int(ratio * total_layers)
            indices.append(min(idx, total_layers - 1))
        
        indices[0] = 0  # 确保包含第一层
        indices[-1] = total_layers - 1  # 确保包含最后一层
        
        return sorted(set(indices))
    
    def _logarithmic_sampling(self, total_layers: int, num_sampled: int) -> List[int]:
        """对数采样"""
        if num_sampled >= total_layers:
            return list(range(total_layers))
        
        indices = []
        for i in range(num_sampled):
            # 对数分布
            log_idx = math.log(1 + i * (math.exp(1) - 1) / (num_sampled - 1))
            idx = int(log_idx * total_layers)
            indices.append(min(idx, total_layers - 1))
        
        indices[0] = 0
        indices[-1] = total_layers - 1
        
        return sorted(set(indices))
    
    def create_transition_layers(self, hidden_size: int, num_transitions: int) -> nn.ModuleList:
        """创建过渡层来弥补语义断层"""
        transitions = nn.ModuleList()
        
        for i in range(num_transitions):
            transition_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            transitions.append(transition_layer)
        
        return transitions
    
    def analyze_sampling_quality(self, total_layers: int, sampled_indices: List[int]) -> Dict[str, Any]:
        """分析采样质量"""
        coverage = len(sampled_indices) / total_layers
        spacing = []
        
        for i in range(1, len(sampled_indices)):
            spacing.append(sampled_indices[i] - sampled_indices[i-1])
        
        analysis = {
            'total_layers': total_layers,
            'sampled_layers': len(sampled_indices),
            'coverage_ratio': coverage,
            'min_spacing': min(spacing) if spacing else 0,
            'max_spacing': max(spacing) if spacing else 0,
            'avg_spacing': sum(spacing) / len(spacing) if spacing else 0,
            'indices': sampled_indices
        }
        
        return analysis

