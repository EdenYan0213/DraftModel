"""
知识增强的草稿模型
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .cross_attention import VectorBasedCrossAttention
from .knowledge_cache import KnowledgeCacheManager


class EnhancedDraftLayer(nn.Module):
    """增强的草稿层 - 使用基于向量的交叉注意力"""
    
    def __init__(self, original_layer: nn.Module, hidden_size: int, num_heads: int,
                 use_cross_attention: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        
        # 复制原始层的组件
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        
        # 交叉注意力（基于向量）
        if use_cross_attention:
            self.cross_attention = VectorBasedCrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads
            )
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                knowledge_vectors: Optional[torch.Tensor] = None,
                answer_start_idx: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """前向传播"""
        # Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        self_attn_outputs = self.self_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = self_attn_outputs[0] if isinstance(self_attn_outputs, tuple) else self_attn_outputs
        hidden_states = residual + hidden_states
        
        # Cross-Attention (如果提供了knowledge_vectors)
        if self.use_cross_attention and knowledge_vectors is not None:
            hidden_states = self.cross_attention(
                query_vectors=hidden_states,
                knowledge_vectors=knowledge_vectors,
                attention_mask=attention_mask,
                answer_start_idx=answer_start_idx
            )
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_outputs = self.mlp(hidden_states)
        mlp_hidden_states = mlp_outputs[0] if isinstance(mlp_outputs, tuple) else mlp_outputs
        hidden_states = residual + mlp_hidden_states
        
        return hidden_states


class Qwen3DraftModel(nn.Module):
    """Qwen3草稿模型 - 使用知识增强机制"""
    
    def __init__(self, config: Dict[str, Any], target_model: nn.Module,
                 knowledge_cache_manager: Optional[KnowledgeCacheManager] = None):
        super().__init__()
        self.config = config
        self.target_model = target_model
        self.knowledge_cache_manager = knowledge_cache_manager
        
        base_config = config['base_model']
        draft_config = config['draft_model']
        
        self.hidden_size = base_config['hidden_size']
        self.num_heads = base_config['num_attention_heads']
        self.num_sampled_layers = draft_config['num_sampled_layers']
        
        # 层采样（选择前N层）
        total_layers = len(target_model.model.layers)
        self.sampled_indices = list(range(min(self.num_sampled_layers, total_layers)))
        
        print(f"从 {total_layers} 层中采样 {len(self.sampled_indices)} 层: {self.sampled_indices}")
        
        # 创建增强层
        self.enhanced_layers = nn.ModuleList()
        for idx in self.sampled_indices:
            original_layer = target_model.model.layers[idx]
            enhanced_layer = EnhancedDraftLayer(
                original_layer=original_layer,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                use_cross_attention=True
            )
            self.enhanced_layers.append(enhanced_layer)
        
        # 复制其他组件
        self.embed_tokens = target_model.model.embed_tokens
        self.norm = target_model.model.norm
        self.lm_head = target_model.lm_head
        
        # 参数统计
        self._print_parameter_count()
    
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
                retrieve_knowledge: bool = True,
                query_text: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """前向传播"""
        # 嵌入层
        hidden_states = self.embed_tokens(input_ids)
        
        # 检索知识（如果启用）
        knowledge_vectors = None
        answer_start_idx = None
        
        if retrieve_knowledge and self.knowledge_cache_manager is not None:
            if query_text is None:
                try:
                    query_text = self.target_model.config.tokenizer.decode(input_ids[0][:20])
                except:
                    query_text = None
            
            if query_text:
                # 获取当前设备，确保知识向量在正确的设备上
                device = hidden_states.device
                result = self.knowledge_cache_manager.retrieve_by_similarity(
                    query=query_text, 
                    top_k=1,
                    device=device  # 直接指定设备，避免后续设备不匹配
                )
                if result is not None:
                    knowledge_vectors, answer_start_idx = result
                    # 扩展batch维度
                    if knowledge_vectors.dim() == 2:
                        knowledge_vectors = knowledge_vectors.unsqueeze(0)
                    batch_size = hidden_states.shape[0]
                    if knowledge_vectors.shape[0] == 1:
                        knowledge_vectors = knowledge_vectors.expand(batch_size, -1, -1)
        
        # 计算position_embeddings
        if 'position_ids' not in kwargs:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            kwargs['position_ids'] = position_ids
        
        if 'position_embeddings' not in kwargs and hasattr(self.target_model, 'model') and hasattr(self.target_model.model, 'rotary_emb'):
            position_embeddings = self.target_model.model.rotary_emb(hidden_states, kwargs['position_ids'])
            kwargs['position_embeddings'] = position_embeddings
        
        # 修复attention_mask的dtype
        if attention_mask is not None and attention_mask.dtype == torch.long:
            attention_mask = None
        
        # 逐层处理
        for layer in self.enhanced_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                knowledge_vectors=knowledge_vectors,
                answer_start_idx=answer_start_idx,
                **kwargs
            )
        
        # 最终归一化和输出
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }

