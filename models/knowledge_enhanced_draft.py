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
    
    def __init__(self, hidden_size: int, num_heads: int, cache_dim: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
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
        
        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                knowledge_keys: Optional[torch.Tensor] = None,
                knowledge_values: Optional[torch.Tensor] = None,
                knowledge_cache: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            
            # 使用缓存的KV矩阵进行交叉注意力
            # Q = hidden_states (Self-Attention的输出)
            # K, V = knowledge_keys, knowledge_values (缓存的KV)
            knowledge_output, _ = self.knowledge_attn(
                query=hidden_states,  # Q: 当前层的输出
                key=knowledge_keys,   # K: 缓存的Key
                value=knowledge_values, # V: 缓存的Value
                key_padding_mask=None,
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
        
        # 门控融合
        combined = torch.cat([hidden_states, knowledge_output], dim=-1)
        gate_weights = self.fusion_gate(combined)
        enhanced_states = gate_weights * hidden_states + (1 - gate_weights) * knowledge_output
        
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
            self.knowledge_enhancer = KnowledgeEnhancedCrossAttention(
                hidden_size, num_heads
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
                    cross_attn_output = self.knowledge_enhancer(
                        hidden_states,
                        knowledge_keys=knowledge_keys,
                        knowledge_values=knowledge_values,
                        attention_mask=attention_mask
                    )
                elif isinstance(knowledge_cache, torch.Tensor):
                    # 压缩的缓存向量
                    cross_attn_output = self.knowledge_enhancer(
                        hidden_states,
                        knowledge_cache=knowledge_cache,
                        attention_mask=attention_mask
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
            
            # 检索KV矩阵
            kv_result = self.knowledge_cache.retrieve(query_text)
            if kv_result is not None:
                knowledge_keys, knowledge_values = kv_result
                # 移动到正确的设备
                device = hidden_states.device
                knowledge_keys = knowledge_keys.to(device)
                knowledge_values = knowledge_values.to(device)
                current_knowledge = (knowledge_keys, knowledge_values)
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
            'sampled_indices': self.sampled_indices
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

