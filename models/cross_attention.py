"""
基于向量的交叉注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class VectorBasedCrossAttention(nn.Module):
    """
    基于向量的交叉注意力机制
    
    输入：
    - query_vectors: (batch_size, query_seq_len, hidden_size) - 问题序列向量
    - knowledge_vectors: (batch_size, knowledge_seq_len, hidden_size) - 检索到的相似序列向量
    
    从向量投影得到QKV矩阵，然后进行交叉注意力计算
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        
        # QKV投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self,
                query_vectors: torch.Tensor,
                knowledge_vectors: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                answer_start_idx: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query_vectors: (batch_size, query_seq_len, hidden_size) - 问题序列向量
            knowledge_vectors: (batch_size, knowledge_seq_len, hidden_size) - 检索到的相似序列向量
            attention_mask: (batch_size, query_seq_len) - 注意力掩码（可选）
            answer_start_idx: 答案在knowledge_vectors中的起始位置（可选，用于mask问题部分）
        
        Returns:
            output: (batch_size, query_seq_len, hidden_size) - 交叉注意力输出
        """
        batch_size, query_seq_len, hidden_size = query_vectors.shape
        knowledge_seq_len = knowledge_vectors.shape[1]
        
        # 确保输入数据类型与层参数类型匹配
        # 如果层参数是float32但输入是float16，或者反之，需要转换
        layer_dtype = next(self.q_proj.parameters()).dtype
        if query_vectors.dtype != layer_dtype:
            query_vectors = query_vectors.to(dtype=layer_dtype)
        if knowledge_vectors.dtype != layer_dtype:
            knowledge_vectors = knowledge_vectors.to(dtype=layer_dtype)
        
        # 从向量投影得到QKV
        Q = self.q_proj(query_vectors)
        K = self.k_proj(knowledge_vectors)
        V = self.v_proj(knowledge_vectors)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, knowledge_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, knowledge_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 如果提供了answer_start_idx，mask掉问题部分（只关注答案部分）
        if answer_start_idx is not None and answer_start_idx > 0:
            knowledge_mask = torch.zeros(batch_size, 1, 1, knowledge_seq_len, 
                                       device=scores.device, dtype=scores.dtype)
            knowledge_mask[:, :, :, :answer_start_idx] = float('-inf')
            scores = scores + knowledge_mask
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, query_seq_len, hidden_size)
        
        # 输出投影和残差连接
        output = self.out_proj(attn_output)
        output = self.layer_norm(query_vectors + output)
        
        return output

