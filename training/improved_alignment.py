"""
改进的对齐机制 - 提高草稿模型与基础模型的输出对齐度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def compute_improved_acceptance_loss(draft_logits: torch.Tensor,
                                     target_logits: torch.Tensor,
                                     temperature: float = 1.0) -> torch.Tensor:
    """
    改进的接受概率损失 - 最大化目标模型对草稿模型预测token的概率
    
    方法1: 直接最大化目标模型对草稿模型argmax预测的概率
    方法2: 使用温度缩放让分布更尖锐
    方法3: 使用Top-k对齐（只关注Top-k token）
    
    Args:
        draft_logits: 草稿模型的logits (batch, seq, vocab)
        target_logits: 目标模型的logits (batch, seq, vocab)
        temperature: 温度参数，用于缩放分布
    
    Returns:
        loss: 接受概率损失
    """
    # 应用温度缩放
    draft_logits_scaled = draft_logits / temperature
    target_logits_scaled = target_logits / temperature
    
    # 计算概率分布
    draft_probs = F.softmax(draft_logits_scaled, dim=-1)  # (batch, seq, vocab)
    target_probs = F.softmax(target_logits_scaled, dim=-1)  # (batch, seq, vocab)
    
    # 方法1: 直接最大化目标模型对草稿模型argmax预测的概率
    draft_predictions = torch.argmax(draft_logits_scaled, dim=-1)  # (batch, seq)
    batch_size, seq_len = draft_predictions.shape
    
    # 获取目标模型对草稿模型预测token的概率
    target_probs_for_draft = torch.gather(
        target_probs, 
        dim=-1, 
        index=draft_predictions.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq)
    
    # 接受概率损失：最小化负对数概率（最大化概率）
    acceptance_loss = -torch.mean(torch.log(target_probs_for_draft + 1e-8))
    
    return acceptance_loss


def compute_topk_alignment_loss(draft_logits: torch.Tensor,
                                 target_logits: torch.Tensor,
                                 top_k: int = 10,
                                 temperature: float = 1.0) -> torch.Tensor:
    """
    Top-k对齐损失 - 让草稿模型的Top-k预测与目标模型的Top-k预测对齐
    
    Args:
        draft_logits: 草稿模型的logits (batch, seq, vocab)
        target_logits: 目标模型的logits (batch, seq, vocab)
        top_k: Top-k token数量
        temperature: 温度参数
    
    Returns:
        loss: Top-k对齐损失
    """
    # 应用温度缩放
    draft_logits_scaled = draft_logits / temperature
    target_logits_scaled = target_logits / temperature
    
    # 计算概率分布
    draft_probs = F.softmax(draft_logits_scaled, dim=-1)
    target_probs = F.softmax(target_logits_scaled, dim=-1)
    
    # 获取Top-k token和概率
    draft_topk_probs, draft_topk_indices = torch.topk(draft_probs, top_k, dim=-1)
    target_topk_probs, target_topk_indices = torch.topk(target_probs, top_k, dim=-1)
    
    # 计算Top-k token的重叠度
    batch_size, seq_len, vocab_size = draft_probs.shape
    overlap_loss = torch.tensor(0.0, device=draft_probs.device)
    
    for b in range(batch_size):
        for s in range(seq_len):
            draft_set = set(draft_topk_indices[b, s].cpu().numpy())
            target_set = set(target_topk_indices[b, s].cpu().numpy())
            
            # 计算Jaccard相似度
            intersection = len(draft_set & target_set)
            union = len(draft_set | target_set)
            jaccard = intersection / union if union > 0 else 0.0
            
            # 损失：1 - Jaccard相似度（最小化）
            overlap_loss += (1.0 - jaccard)
    
    overlap_loss = overlap_loss / (batch_size * seq_len)
    
    # 同时最大化目标模型对草稿模型Top-k token的概率
    topk_alignment_loss = torch.tensor(0.0, device=draft_probs.device)
    
    for b in range(batch_size):
        for s in range(seq_len):
            draft_topk = draft_topk_indices[b, s]  # (top_k,)
            target_probs_for_draft_topk = target_probs[b, s, draft_topk]  # (top_k,)
            
            # 最大化目标模型对草稿模型Top-k token的概率
            topk_alignment_loss -= torch.mean(torch.log(target_probs_for_draft_topk + 1e-8))
    
    topk_alignment_loss = topk_alignment_loss / (batch_size * seq_len)
    
    # 组合损失
    total_loss = overlap_loss + topk_alignment_loss
    
    return total_loss


def compute_sequence_alignment_loss(draft_logits: torch.Tensor,
                                    target_logits: torch.Tensor,
                                    sequence_length: int = 5,
                                    temperature: float = 1.0) -> torch.Tensor:
    """
    序列级对齐损失 - 对齐整个序列的预测，而不仅仅是单个token
    
    Args:
        draft_logits: 草稿模型的logits (batch, seq, vocab)
        target_logits: 目标模型的logits (batch, seq, vocab)
        sequence_length: 序列长度
        temperature: 温度参数
    
    Returns:
        loss: 序列对齐损失
    """
    # 应用温度缩放
    draft_logits_scaled = draft_logits / temperature
    target_logits_scaled = target_logits / temperature
    
    # 计算概率分布
    draft_probs = F.softmax(draft_logits_scaled, dim=-1)
    target_probs = F.softmax(target_logits_scaled, dim=-1)
    
    batch_size, seq_len, vocab_size = draft_probs.shape
    
    # 对每个序列位置，计算草稿模型和目标模型的分布差异
    sequence_loss = torch.tensor(0.0, device=draft_probs.device)
    
    # 使用KL散度计算序列级对齐
    for i in range(min(sequence_length, seq_len)):
        draft_dist = draft_probs[:, i, :]  # (batch, vocab)
        target_dist = target_probs[:, i, :]  # (batch, vocab)
        
        # KL散度：KL(draft || target)
        kl_div = F.kl_div(
            F.log_softmax(draft_logits_scaled[:, i, :], dim=-1),
            target_dist,
            reduction='none'
        ).sum(dim=-1)  # (batch,)
        
        sequence_loss += kl_div.mean()
    
    sequence_loss = sequence_loss / min(sequence_length, seq_len)
    
    return sequence_loss


def compute_greedy_alignment_loss(draft_logits: torch.Tensor,
                                  target_logits: torch.Tensor,
                                  temperature: float = 1.0) -> torch.Tensor:
    """
    贪心对齐损失 - 让草稿模型的贪心预测与目标模型的贪心预测对齐
    
    这是最直接的对齐方法：最大化目标模型对草稿模型贪心预测的概率
    
    Args:
        draft_logits: 草稿模型的logits (batch, seq, vocab)
        target_logits: 目标模型的logits (batch, seq, vocab)
        temperature: 温度参数
    
    Returns:
        loss: 贪心对齐损失
    """
    # 应用温度缩放
    draft_logits_scaled = draft_logits / temperature
    target_logits_scaled = target_logits / temperature
    
    # 草稿模型的贪心预测（argmax）
    draft_predictions = torch.argmax(draft_logits_scaled, dim=-1)  # (batch, seq)
    
    # 目标模型的概率分布
    target_probs = F.softmax(target_logits_scaled, dim=-1)  # (batch, seq, vocab)
    
    # 获取目标模型对草稿模型预测token的概率
    batch_size, seq_len = draft_predictions.shape
    target_probs_for_draft = torch.gather(
        target_probs,
        dim=-1,
        index=draft_predictions.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq)
    
    # 贪心对齐损失：最小化负对数概率（最大化概率）
    greedy_loss = -torch.mean(torch.log(target_probs_for_draft + 1e-8))
    
    return greedy_loss


def compute_combined_alignment_loss(draft_logits: torch.Tensor,
                                    target_logits: torch.Tensor,
                                    weights: Optional[Dict[str, float]] = None,
                                    temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    组合对齐损失 - 结合多种对齐方法
    
    Args:
        draft_logits: 草稿模型的logits (batch, seq, vocab)
        target_logits: 目标模型的logits (batch, seq, vocab)
        weights: 各损失权重 {'greedy': 0.5, 'topk': 0.3, 'sequence': 0.2}
        temperature: 温度参数
    
    Returns:
        total_loss: 总损失
        loss_dict: 各损失分量
    """
    if weights is None:
        weights = {
            'greedy': 0.5,
            'topk': 0.3,
            'sequence': 0.2
        }
    
    # 计算各种对齐损失
    greedy_loss = compute_greedy_alignment_loss(draft_logits, target_logits, temperature)
    topk_loss = compute_topk_alignment_loss(draft_logits, target_logits, top_k=10, temperature=temperature)
    sequence_loss = compute_sequence_alignment_loss(draft_logits, target_logits, sequence_length=5, temperature=temperature)
    
    # 组合损失
    total_loss = (
        weights.get('greedy', 0.0) * greedy_loss +
        weights.get('topk', 0.0) * topk_loss +
        weights.get('sequence', 0.0) * sequence_loss
    )
    
    loss_dict = {
        'greedy': greedy_loss,
        'topk': topk_loss,
        'sequence': sequence_loss,
        'total': total_loss
    }
    
    return total_loss, loss_dict


