"""
注意力引导的分布对齐模块

利用注意力权重指导草稿模型输出与基础模型输出的对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def attention_guided_alignment(draft_logits: torch.Tensor, 
                               base_logits: torch.Tensor, 
                               draft_attention: Optional[torch.Tensor] = None,
                               base_attention: Optional[torch.Tensor] = None,
                               attention_threshold: float = 0.1,
                               alpha: float = 0.5) -> torch.Tensor:
    """
    利用注意力权重指导分布对齐
    对高注意力权重的token进行更强的对齐
    
    Args:
        draft_logits: 草稿模型的logits, shape: [batch_size, seq_len, vocab_size]
        base_logits: 基础模型的logits, shape: [batch_size, seq_len, vocab_size]
        draft_attention: 草稿模型的注意力权重 (可选)
        base_attention: 基础模型的注意力权重 (可选)
        attention_threshold: 注意力阈值
        alpha: 基础混合权重 (如果没有注意力权重则使用)
        
    Returns:
        aligned_log_probs: 对齐后的log概率, shape: [batch_size, seq_len, vocab_size]
    """
    batch_size, seq_len, vocab_size = draft_logits.shape
    
    # 计算概率分布
    draft_probs = F.softmax(draft_logits, dim=-1)
    base_probs = F.softmax(base_logits, dim=-1)
    
    # 如果有attention权重，使用动态混合
    if base_attention is not None:
        # 计算token级别的注意力重要性
        # base_attention shape可能是: [batch_size, num_heads, seq_len, seq_len]
        # 取最后一层的attention，并对head维度平均
        if base_attention.dim() == 4:
            # 平均所有head
            token_importance = base_attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
            # 对source维度求和，得到每个target token的重要性
            token_importance = token_importance.sum(dim=-1)  # [batch_size, seq_len]
        else:
            # 如果已经是2D的，直接使用
            token_importance = base_attention.mean(dim=1) if base_attention.dim() == 3 else base_attention
        
        # 归一化重要性分数
        token_importance = token_importance / (token_importance.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 扩展维度以匹配logits
        importance_weights = token_importance.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # 动态alpha：高注意力位置使用更高的基础模型权重
        # 映射到[alpha/2, alpha]范围
        dynamic_alpha = importance_weights * alpha
        
        # 应用动态混合
        mixed_probs = (1 - dynamic_alpha) * draft_probs + dynamic_alpha * base_probs
    else:
        # 没有attention权重，使用固定混合
        mixed_probs = (1 - alpha) * draft_probs + alpha * base_probs
    
    # 返回log概率
    return torch.log(mixed_probs + 1e-8)


def compute_alignment_loss(draft_logits: torch.Tensor,
                           base_logits: torch.Tensor,
                           draft_attention: Optional[torch.Tensor] = None,
                           base_attention: Optional[torch.Tensor] = None,
                           loss_type: str = 'kl') -> torch.Tensor:
    """
    计算对齐损失
    
    Args:
        draft_logits: 草稿模型的logits
        base_logits: 基础模型的logits  
        draft_attention: 草稿模型的注意力权重
        base_attention: 基础模型的注意力权重
        loss_type: 损失类型，'kl'或'mse'
        
    Returns:
        loss: 对齐损失
    """
    draft_probs = F.softmax(draft_logits, dim=-1)
    base_probs = F.softmax(base_logits, dim=-1)
    
    if loss_type == 'kl':
        # KL散度损失
        kl_div = F.kl_div(
            F.log_softmax(draft_logits, dim=-1),
            base_probs,
            reduction='none'
        ).sum(dim=-1)  # [batch_size, seq_len]
        
        # 如果有attention权重，加权KL散度
        if base_attention is not None:
            if base_attention.dim() == 4:
                token_importance = base_attention.mean(dim=1).sum(dim=-1)
            else:
                token_importance = base_attention.mean(dim=1) if base_attention.dim() == 3 else base_attention
            
            token_importance = token_importance / (token_importance.sum(dim=-1, keepdim=True) + 1e-8)
            kl_div = kl_div * token_importance
        
        loss = kl_div.mean()
    
    elif loss_type == 'mse':
        # MSE损失（概率空间）
        mse_loss = F.mse_loss(draft_probs, base_probs, reduction='none').mean(dim=-1)
        
        # 如果有attention权重，加权MSE
        if base_attention is not None:
            if base_attention.dim() == 4:
                token_importance = base_attention.mean(dim=1).sum(dim=-1)
            else:
                token_importance = base_attention.mean(dim=1) if base_attention.dim() == 3 else base_attention
            
            token_importance = token_importance / (token_importance.sum(dim=-1, keepdim=True) + 1e-8)
            mse_loss = mse_loss * token_importance
        
        loss = mse_loss.mean()
    
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")
    
    return loss


class AttentionAlignmentModule(nn.Module):
    """注意力对齐模块"""
    
    def __init__(self, alpha: float = 0.5, loss_type: str = 'kl'):
        super().__init__()
        self.alpha = alpha
        self.loss_type = loss_type
    
    def forward(self, 
                draft_output: dict,
                base_output: dict,
                compute_loss: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            draft_output: 草稿模型输出字典，包含'logits'和可选的'attentions'
            base_output: 基础模型输出字典，包含'logits'和可选的'attentions'
            compute_loss: 是否计算损失
            
        Returns:
            aligned_logits: 对齐后的logits
            loss: 对齐损失（如果compute_loss=True）
        """
        draft_logits = draft_output['logits'] if isinstance(draft_output, dict) else draft_output.logits
        base_logits = base_output['logits'] if isinstance(base_output, dict) else base_output.logits
        
        draft_attention = draft_output.get('attentions') if isinstance(draft_output, dict) else getattr(draft_output, 'attentions', None)
        base_attention = base_output.get('attentions') if isinstance(base_output, dict) else getattr(base_output, 'attentions', None)
        
        # 计算对齐后的logits
        aligned_log_probs = attention_guided_alignment(
            draft_logits,
            base_logits,
            draft_attention,
            base_attention,
            alpha=self.alpha
        )
        
        # 计算损失
        loss = None
        if compute_loss:
            loss = compute_alignment_loss(
                draft_logits,
                base_logits,
                draft_attention,
                base_attention,
                loss_type=self.loss_type
            )
        
        return aligned_log_probs, loss


def align_with_attention(draft_output: dict, 
                         base_output: dict,
                         alpha: float = 0.3) -> torch.Tensor:
    """
    使用注意力权重对齐草稿模型和基础模型的输出
    
    Args:
        draft_output: 草稿模型输出
        base_output: 基础模型输出
        alpha: 混合权重
        
    Returns:
        aligned_log_probs: 对齐后的log概率
    """
    draft_logits = draft_output['logits'] if isinstance(draft_output, dict) else draft_output.logits
    base_logits = base_output['logits'] if isinstance(base_output, dict) else base_output.logits
    
    draft_attention = draft_output.get('attentions') if isinstance(draft_output, dict) else getattr(draft_output, 'attentions', None)
    base_attention = base_output.get('attentions') if isinstance(base_output, dict) else getattr(base_output, 'attentions', None)
    
    # 提取最后一层的attention（如果有）
    if draft_attention is not None and isinstance(draft_attention, (list, tuple)):
        draft_attention = draft_attention[-1]
    if base_attention is not None and isinstance(base_attention, (list, tuple)):
        base_attention = base_attention[-1]
    
    return attention_guided_alignment(
        draft_logits,
        base_logits,
        draft_attention,
        base_attention,
        alpha=alpha
    )

