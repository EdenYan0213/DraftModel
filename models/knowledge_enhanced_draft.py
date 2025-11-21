import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM
from typing import Optional, Dict, Any, List, Tuple, Union
import math
from .layer_sampler import LayerSampler

class KnowledgeEnhancedCrossAttention(nn.Module):
    """知识增强的交叉注意力层 - 使用缓存的KV矩阵"""
    
    def __init__(self, hidden_size: int, num_heads: int, cache_dim: int = 512, 
                 question_weight: float = 0.3, answer_weight: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.question_weight = question_weight  # 问题部分权重
        self.answer_weight = answer_weight      # 答案部分权重
        
        # 知识交叉注意力
        self.knowledge_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 知识投影层（用于处理压缩缓存）
        self.kv_projection = nn.Sequential(
            nn.Linear(cache_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2)
        )
        
        # 门控融合机制（改进版：考虑知识置信度）
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 知识置信度网络（评估知识输出的可靠性）
        self.knowledge_confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                knowledge_keys: Optional[torch.Tensor] = None,
                knowledge_values: Optional[torch.Tensor] = None,
                knowledge_cache: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                knowledge_similarity: Optional[float] = None,
                force_knowledge_weight: Optional[float] = None,
                current_answer_length: Optional[int] = None,
                answer_start_idx: Optional[int] = None,
                query_length: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - Self-Attention的输出
            knowledge_keys: (num_heads, cache_seq_len, head_dim) 或 (batch_size, cache_seq_len, hidden_size) - 缓存的Key矩阵
            knowledge_values: (num_heads, cache_seq_len, head_dim) 或 (batch_size, cache_seq_len, hidden_size) - 缓存的Value矩阵
            knowledge_cache: (batch_size, cache_dim) - 压缩的缓存（备用）
            attention_mask: (batch_size, seq_len)
        
        Returns:
            enhanced_states: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 优先使用直接的KV矩阵
        if knowledge_keys is not None and knowledge_values is not None:
            # 处理KV矩阵的形状
            original_k_shape = knowledge_keys.shape
            original_v_shape = knowledge_values.shape
            
            if knowledge_keys.dim() == 3:
                # 判断是 (num_heads, seq_len, head_dim) 还是 (batch_size, seq_len, hidden_size)
                # 通过检查最后一个维度：head_dim通常较小(64-128)，hidden_size较大(1024)
                last_dim = knowledge_keys.size(-1)
                
                if last_dim < hidden_size // 2:
                    # 最后一个维度较小，可能是 (num_heads, cache_seq_len, head_dim) 格式
                    # 需要转换为 (batch_size, cache_seq_len, hidden_size)
                    cache_seq_len = knowledge_keys.size(1)
                    head_dim = knowledge_keys.size(2)
                    
                    # 检查第一个维度是否是num_heads（或者可能是batch_size，但head_dim不对）
                    # 如果 num_heads * head_dim == hidden_size，说明是多头格式
                    if knowledge_keys.size(0) * head_dim == hidden_size or \
                       (knowledge_keys.size(0) == self.num_heads and self.num_heads * head_dim == hidden_size):
                        # 转置并重塑 (num_heads, seq_len, head_dim) -> (seq_len, num_heads, head_dim) -> (seq_len, hidden_size)
                        k_reshaped = knowledge_keys.transpose(0, 1).contiguous()  # (seq_len, num_heads, head_dim)
                        k_reshaped = k_reshaped.view(cache_seq_len, knowledge_keys.size(0) * head_dim)  # (seq_len, hidden_size)
                        
                        v_reshaped = knowledge_values.transpose(0, 1).contiguous()  # (seq_len, num_heads, head_dim)
                        v_reshaped = v_reshaped.view(cache_seq_len, knowledge_values.size(0) * head_dim)  # (seq_len, hidden_size)
                        
                        # 扩展batch维度
                        knowledge_keys = k_reshaped.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, hidden_size)
                        knowledge_values = v_reshaped.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, hidden_size)
                    else:
                        # 可能是其他格式，尝试直接reshape
                        # 假设第一个维度是num_heads
                        num_heads_in_cache = knowledge_keys.size(0)
                        if num_heads_in_cache * head_dim == hidden_size:
                            k_reshaped = knowledge_keys.transpose(0, 1).contiguous().view(cache_seq_len, hidden_size)
                            v_reshaped = knowledge_values.transpose(0, 1).contiguous().view(cache_seq_len, hidden_size)
                            knowledge_keys = k_reshaped.unsqueeze(0).expand(batch_size, -1, -1)
                            knowledge_values = v_reshaped.unsqueeze(0).expand(batch_size, -1, -1)
                        else:
                            raise ValueError(f"无法处理的KV形状: {knowledge_keys.shape}, head_dim={head_dim}, hidden_size={hidden_size}, num_heads={num_heads_in_cache}")
                    
                elif last_dim == hidden_size:
                    # 已经是 (batch_size, seq_len, hidden_size) 或 (seq_len, hidden_size)
                    if knowledge_keys.size(0) != batch_size:
                        # (seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
                        knowledge_keys = knowledge_keys.unsqueeze(0).expand(batch_size, -1, -1)
                        knowledge_values = knowledge_values.unsqueeze(0).expand(batch_size, -1, -1)
                    # 否则已经是正确的形状，直接使用
                else:
                    raise ValueError(f"无法处理的KV形状: {knowledge_keys.shape}, head_dim={self.head_dim}, hidden_size={hidden_size}")
            elif knowledge_keys.dim() == 2:
                # (cache_seq_len, hidden_size) -> (batch_size, cache_seq_len, hidden_size)
                if knowledge_keys.size(-1) == hidden_size:
                    knowledge_keys = knowledge_keys.unsqueeze(0).expand(batch_size, -1, -1)
                    knowledge_values = knowledge_values.unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    raise ValueError(f"无法处理knowledge_keys形状: {knowledge_keys.shape}, hidden_size={hidden_size}")
            else:
                # 未知形状，尝试reshape
                if knowledge_keys.size(-1) == hidden_size:
                    knowledge_keys = knowledge_keys.view(batch_size, -1, hidden_size)
                    knowledge_values = knowledge_values.view(batch_size, -1, hidden_size)
                else:
                    raise ValueError(f"无法处理knowledge_keys形状: {knowledge_keys.shape}")
            
            # 优化：使用问题+答案的完整KV，使用动态加权mask策略
            # 这样Query和Key的语义上下文匹配（都包含问题）
            aligned_keys = knowledge_keys
            aligned_values = knowledge_values
            
            # 优化Mask策略2：使用动态加权mask
            # 根据生成阶段动态调整问题部分权重：
            # - 生成初期（前20%）：问题权重0.5（保留更多上下文）
            # - 生成后期：问题权重0.2（更关注答案）
            # 这样既保留了问题的上下文信息，又重点关注答案部分
            attn_mask = None
            if answer_start_idx is not None and answer_start_idx > 0:
                # 获取KV序列长度
                if knowledge_keys.dim() == 3:
                    # (num_heads, seq_len, head_dim)
                    kv_seq_len = knowledge_keys.shape[1]
                elif knowledge_keys.dim() == 2:
                    # (seq_len, hidden_size)
                    kv_seq_len = knowledge_keys.shape[0]
                elif knowledge_keys.dim() == 4:
                    # (batch, num_heads, seq_len, head_dim)
                    kv_seq_len = knowledge_keys.shape[2]
                else:
                    kv_seq_len = knowledge_keys.shape[-2] if knowledge_keys.dim() > 1 else 1
                
                # 创建加权mask：问题部分权重0.3，答案部分权重1.0
                # 使用attention mask而非key_padding_mask，因为我们需要加权而非完全屏蔽
                if answer_start_idx < kv_seq_len:
                    # 创建attention mask: (seq_len, seq_len)
                    # 对于每个query位置，对key的问题部分应用较低权重
                    # 由于MultiheadAttention不支持直接加权mask，我们使用以下策略：
                    # 1. 在计算attention之前，对问题部分的key进行缩放
                    # 2. 或者使用key_padding_mask但配合权重调整
                    
                    # 优化：基于相似度和生成阶段的动态mask策略
                    # 1. 根据知识相似度确定权重范围
                    # 2. 根据生成阶段在范围内动态调整
                    
                    # 步骤1：根据知识相似度确定权重范围（优化版：更细粒度）
                    if knowledge_similarity is not None:
                        if knowledge_similarity > 0.85:
                            # 极高相似度：更关注答案（0.3 → 0.05）
                            weight_start = 0.3
                            weight_end = 0.05
                            weight_range = weight_start - weight_end  # 0.25
                        elif knowledge_similarity > 0.7:
                            # 高相似度：更关注答案（0.4 → 0.1）
                            weight_start = 0.4
                            weight_end = 0.1
                            weight_range = weight_start - weight_end  # 0.3
                        elif knowledge_similarity > 0.5:
                            # 中等相似度：平衡（0.5 → 0.2）
                            weight_start = 0.5
                            weight_end = 0.2
                            weight_range = weight_start - weight_end  # 0.3
                        elif knowledge_similarity > 0.3:
                            # 低相似度：保留更多问题信息（0.6 → 0.3）
                            weight_start = 0.6
                            weight_end = 0.3
                            weight_range = weight_start - weight_end  # 0.3
                        else:
                            # 极低相似度：保留更多问题信息（0.7 → 0.4）
                            weight_start = 0.7
                            weight_end = 0.4
                            weight_range = weight_start - weight_end  # 0.3
                    else:
                        # 无相似度信息：使用默认范围（0.5 → 0.2）
                        weight_start = 0.5
                        weight_end = 0.2
                        weight_range = weight_start - weight_end  # 0.3
                    
                    # 步骤2：根据生成阶段在范围内动态调整
                    if query_length is not None and query_length > 0 and current_answer_length is not None:
                        # 计算生成比例（0.0到1.0）
                        generation_ratio = min(current_answer_length / max(query_length, 1), 1.0)
                        # 在权重范围内线性衰减
                        question_weight = weight_start - weight_range * generation_ratio
                        question_weight = max(weight_end, min(weight_start, question_weight))  # 限制在范围内
                    else:
                        # 无生成阶段信息：使用起始权重
                        question_weight = weight_start
                    
                    answer_weight = self.answer_weight     # 答案部分权重（可配置）
                    
                    if knowledge_keys.dim() == 3:
                        # (num_heads, seq_len, head_dim)
                        # 对问题部分的key和value进行缩放
                        aligned_keys = knowledge_keys.clone()
                        aligned_values = knowledge_values.clone()
                        aligned_keys[:, :answer_start_idx, :] *= question_weight
                        aligned_values[:, :answer_start_idx, :] *= question_weight
                        # 答案部分保持原样（已经是1.0权重）
                    elif knowledge_keys.dim() == 2:
                        # (seq_len, hidden_size)
                        aligned_keys = knowledge_keys.clone()
                        aligned_values = knowledge_values.clone()
                        aligned_keys[:answer_start_idx, :] *= question_weight
                        aligned_values[:answer_start_idx, :] *= question_weight
                    elif knowledge_keys.dim() == 4:
                        # (batch, num_heads, seq_len, head_dim)
                        aligned_keys = knowledge_keys.clone()
                        aligned_values = knowledge_values.clone()
                        aligned_keys[:, :, :answer_start_idx, :] *= question_weight
                        aligned_values[:, :, :answer_start_idx, :] *= question_weight
                    
                    # 同时使用key_padding_mask进一步降低问题部分的影响
                    # 但设置为False（不屏蔽），因为我们已经在权重上做了处理
                    # 这里可以添加一个轻微的mask来进一步降低问题部分的attention
                    # 创建soft mask：问题部分为True（轻微mask），答案部分为False
                    # 注意：key_padding_mask中True表示需要mask的位置
                    # 但我们可以通过设置一个非常小的mask值来实现"轻微mask"
                    # 由于MultiheadAttention的限制，我们主要依赖权重缩放
                    key_padding_mask = None  # 不使用完全屏蔽，只使用权重缩放
                else:
                    # 如果answer_start_idx >= kv_seq_len，使用完整KV
                    aligned_keys = knowledge_keys
                    aligned_values = knowledge_values
                    key_padding_mask = None
            else:
                # 没有答案起始位置信息，使用完整KV
                aligned_keys = knowledge_keys
                aligned_values = knowledge_values
                key_padding_mask = None
            
            # 使用加权后的KV进行交叉注意力
            # Q = hidden_states (Self-Attention的输出，包含问题+部分生成)
            # K, V = aligned_keys, aligned_values (加权后的完整KV：问题部分权重0.3，答案部分权重1.0)
            knowledge_output, _ = self.knowledge_attn(
                query=hidden_states,  # Q: 当前层的输出（问题+部分生成）
                key=aligned_keys,   # K: 加权后的完整KV（问题部分权重0.3，答案部分权重1.0）
                value=aligned_values, # V: 加权后的完整KV（问题部分权重0.3，答案部分权重1.0）
                key_padding_mask=key_padding_mask,  # 不使用完全屏蔽
                need_weights=False
            )
        
        # 如果没有直接的KV，尝试使用压缩缓存
        elif knowledge_cache is not None:
            # 投影知识缓存到KV空间
            projected_kv = self.kv_projection(knowledge_cache)  # (batch_size, hidden_size * 2)
            
            # 分割为key和value
            k = projected_kv[:, :hidden_size].unsqueeze(1)  # (batch_size, 1, hidden_size)
            v = projected_kv[:, hidden_size:].unsqueeze(1)  # (batch_size, 1, hidden_size)
            
            # 重复以匹配序列长度
            knowledge_keys = k.expand(batch_size, seq_len, hidden_size)
            knowledge_values = v.expand(batch_size, seq_len, hidden_size)
            
            # 交叉注意力
            knowledge_output, _ = self.knowledge_attn(
                query=hidden_states,
                key=knowledge_keys,
                value=knowledge_values,
                key_padding_mask=None,
                need_weights=False
            )
        else:
            # 没有知识缓存，直接返回
            return hidden_states
        
        # 改进的门控融合机制（简化版：主要依赖相似度和强制权重）
        combined = torch.cat([hidden_states, knowledge_output], dim=-1)
        base_gate_weights = self.fusion_gate(combined)  # (batch_size, seq_len, hidden_size)
        
        # 如果强制指定知识权重（推理时使用，优先级最高）
        if force_knowledge_weight is not None:
            # force_knowledge_weight 表示知识的最小权重（0-1）
            # 例如 0.6 表示至少60%使用知识，即 gate_weights 最大为 0.4
            max_original_weight = 1.0 - force_knowledge_weight
            # 直接设置门控权重
            final_gate_weights = torch.clamp(base_gate_weights, max=max_original_weight)
        elif knowledge_similarity is not None:
            # 根据相似度调整门控权重（不使用未训练的置信度网络）
            # 相似度高时，更多使用知识（降低原始输出权重）
            similarity_factor = torch.tensor(knowledge_similarity, device=base_gate_weights.device, dtype=base_gate_weights.dtype)
            # 相似度越高，原始输出权重越低（但保持至少30%的原始输出）
            # similarity_factor * 0.6 表示最多降低60%的原始权重
            final_gate_weights = base_gate_weights * (0.3 + 0.4 * (1 - similarity_factor))
        else:
            # 没有相似度信息，使用基础门控权重
            # 但稍微偏向知识（降低原始输出权重10%）
            final_gate_weights = base_gate_weights * 0.9
        
        # 融合：更保守的策略，主要依赖原始输出，知识增强作为辅助
        # 修改：增加原始输出的权重（从默认的gate_weights改为至少0.7），减少知识输出的权重
        # 这样可以提高与目标模型的对齐，因为目标模型不使用知识增强
        conservative_gate = torch.clamp(final_gate_weights, min=0.7, max=1.0)  # 至少70%使用原始输出
        enhanced_states = conservative_gate * hidden_states + (1 - conservative_gate) * knowledge_output
        
        # 层归一化
        enhanced_states = self.layer_norm(enhanced_states)
        
        return enhanced_states

class EnhancedDraftLayer(nn.Module):
    """增强的草稿层 - 包装原始层并添加知识检索"""
    
    def __init__(self, original_layer: nn.Module, hidden_size: int, num_heads: int, 
                 use_knowledge_enhancement: bool = True, rotary_emb=None):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_knowledge_enhancement = use_knowledge_enhancement
        self.rotary_emb = rotary_emb  # 存储rotary_emb引用
        
        # 复制原始层的属性
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        
        # 知识增强
        if use_knowledge_enhancement:
            # 从配置中获取mask权重（如果可用）
            question_weight = 0.3  # 默认问题部分权重
            answer_weight = 1.0   # 默认答案部分权重
            # 可以从kwargs或config中获取，这里使用默认值
            self.knowledge_enhancer = KnowledgeEnhancedCrossAttention(
                hidden_size, num_heads,
                question_weight=question_weight,
                answer_weight=answer_weight
            )
            # Cross-Attention之后的归一化层（如果Cross-Attention内部有Norm，这个可以移除，但保留以保持一致性）
            self.cross_attention_layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                knowledge_cache: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                **kwargs) -> torch.Tensor:
        """
        前向传播：Self-Attention -> CrossAttention -> MLP
        """
        # Step 1: Input LayerNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Step 2: Self-Attention
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 提取self-attention的输出
        if isinstance(self_attn_outputs, tuple):
            hidden_states = self_attn_outputs[0]
        else:
            hidden_states = self_attn_outputs
        
        # Step 2.1: Self-Attention -> Add & Norm
        hidden_states = residual + hidden_states
        # 注意：Qwen3使用Pre-Norm，所以Self-Attention之前已经Norm过了
        # 但为了保持一致性，这里也可以添加一个Norm（可选）
        # 暂时不添加，保持与原始Qwen3结构一致
        
        # Step 3: Cross-Attention (知识增强) - 在Self-Attention之后，MLP之前
        if self.use_knowledge_enhancement and self.knowledge_enhancer is not None:
            if knowledge_cache is not None:
                # 保存残差连接的输入
                cross_attn_residual = hidden_states
                
                # 处理knowledge_cache
                if isinstance(knowledge_cache, tuple):
                    # 直接提供KV矩阵
                    knowledge_keys, knowledge_values = knowledge_cache
                    # 获取知识相似度（如果可用）
                    knowledge_similarity = getattr(self, '_current_knowledge_similarity', None)
                    # 获取答案起始位置（用于mask）
                    answer_start_idx = getattr(self, '_current_answer_start_idx', None)
                    # 获取当前查询长度（用于动态mask）
                    query_length = getattr(self, '_current_query_length', None)
                    cross_attn_output = self.knowledge_enhancer(
                        hidden_states,
                        knowledge_keys=knowledge_keys,
                        knowledge_values=knowledge_values,
                        attention_mask=attention_mask,
                        knowledge_similarity=knowledge_similarity,
                        force_knowledge_weight=getattr(self, '_force_knowledge_weight', None),
                        answer_start_idx=answer_start_idx,
                        query_length=query_length
                    )
                elif isinstance(knowledge_cache, torch.Tensor):
                    # 压缩的缓存向量
                    knowledge_similarity = getattr(self, '_current_knowledge_similarity', None)
                    cross_attn_output = self.knowledge_enhancer(
                        hidden_states,
                        knowledge_cache=knowledge_cache,
                        attention_mask=attention_mask,
                        knowledge_similarity=knowledge_similarity,
                        force_knowledge_weight=getattr(self, '_force_knowledge_weight', None)
                    )
                
                # Step 3.1: Cross-Attention -> Add & Norm
                # Cross-Attention内部已经有LayerNorm（在门控融合后），
                # 但为了遵循标准的Add & Norm模式，我们在Add之后再Norm一次
                hidden_states = cross_attn_residual + cross_attn_output
                # 添加Norm以确保稳定性（Cross-Attention内部虽然有Norm，但Add之后需要再次Norm）
                hidden_states = self.cross_attention_layernorm(hidden_states)
        
        # Step 4: Post-Attention LayerNorm (用于MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Step 5: MLP/FFN
        mlp_outputs = self.mlp(hidden_states)
        
        # 提取MLP的输出
        if isinstance(mlp_outputs, tuple):
            mlp_hidden_states = mlp_outputs[0]
        else:
            mlp_hidden_states = mlp_outputs
        
        # 残差连接
        hidden_states = residual + mlp_hidden_states
        
        return hidden_states

class Qwen3DraftModel(nn.Module):
    """Qwen3-0.6B 草稿模型"""
    
    def __init__(self, config: Dict[str, Any], target_model: nn.Module, 
                 knowledge_cache_manager=None):
        super().__init__()
        self.config = config
        self.target_model = target_model
        self.knowledge_cache = knowledge_cache_manager
        
        # 模型配置
        base_config = config['base_model']
        draft_config = config['draft_model']
        knowledge_config = config.get('knowledge_enhancement', {})
        
        self.hidden_size = base_config['hidden_size']
        self.num_heads = base_config['num_attention_heads']
        self.num_sampled_layers = draft_config['num_sampled_layers']
        self.use_knowledge = knowledge_config.get('enabled', False)
        
        # 层采样
        self.sampler = LayerSampler(config)
        total_layers = len(target_model.model.layers)
        self.sampled_indices = self.sampler.get_uniform_indices(total_layers)
        
        print(f"从 {total_layers} 层中采样 {len(self.sampled_indices)} 层: {self.sampled_indices}")
        
        # 创建增强层
        self.enhanced_layers = self._create_enhanced_layers()
        
        # 过渡层（用于弥补语义断层）
        if len(self.sampled_indices) > 1:
            self.transition_layers = self.sampler.create_transition_layers(
                self.hidden_size, len(self.sampled_indices) - 1
            )
        else:
            self.transition_layers = None
        
        # 复制其他组件
        # 从目标模型中提取嵌入层、归一化层和语言模型头部组件
        self.embed_tokens = target_model.model.embed_tokens
        self.norm = target_model.model.norm
        self.lm_head = target_model.lm_head
        
        # 参数统计
        self._print_parameter_count()
    
    def _create_enhanced_layers(self) -> nn.ModuleList:
        """创建知识增强层"""
        enhanced_layers = nn.ModuleList()
        
        # 获取rotary_emb旋转位置编码（如果存在）
        rotary_emb = None
        if hasattr(self.target_model, 'model') and hasattr(self.target_model.model, 'rotary_emb'):
            rotary_emb = self.target_model.model.rotary_emb
        elif hasattr(self.target_model, 'rotary_emb'):
            rotary_emb = self.target_model.rotary_emb
        
        for idx in self.sampled_indices:
            original_layer = self.target_model.model.layers[idx]
            enhanced_layer = EnhancedDraftLayer(
                original_layer=original_layer,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                use_knowledge_enhancement=self.use_knowledge,
                rotary_emb=rotary_emb  # 传递rotary_emb
            )
            enhanced_layers.append(enhanced_layer)
        
        return enhanced_layers
    
    def _print_parameter_count(self):
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        target_params = sum(p.numel() for p in self.target_model.parameters())
        
        print(f"草稿模型参数量: {total_params:,}")
        print(f"目标模型参数量: {target_params:,}")
        print(f"参数压缩比: {total_params/target_params:.2%}")
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                knowledge_cache: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                retrieve_knowledge: bool = None,
                query_text: Optional[str] = None,
                **kwargs) -> dict[str, list[int] | Any]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            knowledge_cache: 外部提供的知识缓存，可以是：
                - (keys, values) 元组: 直接的KV矩阵
                - torch.Tensor: 压缩的缓存向量
            retrieve_knowledge: 是否检索知识
            query_text: 查询文本（用于检索知识）
        """
        if retrieve_knowledge is None:
            retrieve_knowledge = self.use_knowledge
        
        # 嵌入层
        hidden_states = self.embed_tokens(input_ids)
        
        # 知识检索
        current_knowledge = knowledge_cache
        if retrieve_knowledge and current_knowledge is None and self.knowledge_cache is not None:
            # 从缓存中检索知识KV矩阵
            if query_text is None:
                # 尝试从input_ids解码（简化实现）
                try:
                    query_text = self.target_model.config.tokenizer.decode(input_ids[0][:10])
                except:
                    query_text = "default_query"
            
            # 检索KV矩阵（同时获取相似度和答案起始位置）
            kv_result = self.knowledge_cache.retrieve(query_text, return_similarity=True)
            if kv_result is not None:
                if len(kv_result) == 4:
                    # 返回 (keys, values, similarity, answer_start_idx)
                    knowledge_keys, knowledge_values, similarity, answer_start_idx = kv_result
                    self._current_knowledge_similarity = similarity
                    self._current_answer_start_idx = answer_start_idx
                elif len(kv_result) == 3:
                    # 可能是 (keys, values, similarity) 或 (keys, values, answer_start_idx)
                    if isinstance(kv_result[2], float):
                        knowledge_keys, knowledge_values, similarity = kv_result
                        self._current_knowledge_similarity = similarity
                        self._current_answer_start_idx = None
                    else:
                        knowledge_keys, knowledge_values, answer_start_idx = kv_result
                        self._current_knowledge_similarity = 0.7
                        self._current_answer_start_idx = answer_start_idx
                else:
                    knowledge_keys, knowledge_values = kv_result
                    self._current_knowledge_similarity = 0.7  # 默认相似度
                    self._current_answer_start_idx = None
                
                # 移动到正确的设备
                device = hidden_states.device
                knowledge_keys = knowledge_keys.to(device)
                knowledge_values = knowledge_values.to(device)
                current_knowledge = (knowledge_keys, knowledge_values)
                
                # 设置强制知识权重（更温和的策略）
                if hasattr(self, '_current_knowledge_similarity'):
                    if self._current_knowledge_similarity > 0.8:
                        self._force_knowledge_weight = 0.6  # 至少60%使用知识（更温和）
                    elif self._current_knowledge_similarity > 0.6:
                        self._force_knowledge_weight = 0.4  # 至少40%使用知识
                    else:
                        self._force_knowledge_weight = None  # 不强制，使用相似度调整
            else:
                # 尝试检索压缩缓存
                compressed = self.knowledge_cache.retrieve_compressed(query_text)
                if compressed is not None:
                    device = hidden_states.device
                    current_knowledge = compressed.to(device)
                    # 扩展以匹配batch_size
                    if current_knowledge.dim() == 1:
                        current_knowledge = current_knowledge.unsqueeze(0).expand(hidden_states.size(0), -1)
        
        # 逐层处理
        # 计算position_embeddings（在Qwen3DraftModel层面计算一次）
        if 'position_ids' not in kwargs:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            kwargs['position_ids'] = position_ids
        
        # 计算position_embeddings（使用与Qwen3Model相同的方式）
        if 'position_embeddings' not in kwargs and hasattr(self.target_model, 'model') and hasattr(self.target_model.model, 'rotary_emb'):
            position_embeddings = self.target_model.model.rotary_emb(hidden_states, kwargs['position_ids'])
            kwargs['position_embeddings'] = position_embeddings
        
        # 修复attention_mask的dtype（转换为None，让模型内部处理）
        # Qwen3需要特殊的mask格式，简单的dtype转换不够
        # 最简单的方法是传递None，让模型使用默认的causal mask
        if attention_mask is not None and attention_mask.dtype == torch.long:
            # 暂时设为None，让模型使用默认的causal mask
            attention_mask = None
        
        # 每层：Self-Attention -> CrossAttention(使用缓存的KV) -> MLP
        # 注意：Cross-Attention在Self-Attention之后、MLP之前执行
        for i, layer in enumerate(self.enhanced_layers):
            # 将相似度和强制权重信息传递给层
            if hasattr(self, '_current_knowledge_similarity'):
                layer._current_knowledge_similarity = self._current_knowledge_similarity
            if hasattr(self, '_force_knowledge_weight'):
                layer._force_knowledge_weight = self._force_knowledge_weight
            
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                knowledge_cache=current_knowledge,  # 传递KV矩阵或压缩缓存
                **kwargs
            )
            
            # 在层之间添加过渡层（除了最后一层）
            if (self.transition_layers is not None and 
                i < len(self.enhanced_layers) - 1):
                hidden_states = self.transition_layers[i](hidden_states)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 输出头
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'sampled_indices': self.sampled_indices,
            'attentions': kwargs.get('output_attentions', False)  # 支持输出attention
        }
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor, 
                 max_new_tokens: int = 5,  # 默认只生成5个token
                 **kwargs) -> torch.Tensor:
        """简化生成函数 - 默认生成5个token"""
        current_input = input_ids
        generated = []
        
        # 限制最大生成token数
        max_new_tokens = min(max_new_tokens, 5)  # 最多5个token
        
        for i in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(current_input, **kwargs)
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 采样下一个token（贪心解码）
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_token)
            
            # 更新输入
            current_input = torch.cat([current_input, next_token], dim=1)
            
            # 简单的停止条件
            if hasattr(self.target_model, 'config') and hasattr(self.target_model.config, 'eos_token_id'):
                if next_token.item() == self.target_model.config.eos_token_id:
                    break
        
        return torch.cat(generated, dim=1) if generated else input_ids
    
    def get_sampling_analysis(self) -> Dict[str, Any]:
        """获取采样分析"""
        total_layers = len(self.target_model.model.layers)
        return self.sampler.analyze_sampling_quality(total_layers, self.sampled_indices)

