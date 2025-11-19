"""
简单的草稿模型 - 只使用基础模型的前3层，在每层之间加上交叉注意力机制
不需要训练，直接使用基础模型的权重
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union


class SimpleCrossAttention(nn.Module):
    """简单的交叉注意力层 - 用于层间知识传递"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0  # 推理时不需要dropout
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self,
                hidden_states: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - 当前层的输出
            context: (batch_size, context_len, hidden_size) - 上下文（可选，如果为None则使用hidden_states）
            attention_mask: (batch_size, seq_len)
        
        Returns:
            enhanced_states: (batch_size, seq_len, hidden_size)
        """
        # 如果没有提供context，使用hidden_states作为context（自注意力）
        if context is None:
            context = hidden_states
        
        # 交叉注意力：Q来自hidden_states，K和V来自context
        attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=context,
            value=context,
            key_padding_mask=None,
            need_weights=False
        )
        
        # 残差连接
        enhanced_states = hidden_states + attn_output
        
        # 层归一化
        enhanced_states = self.layer_norm(enhanced_states)
        
        return enhanced_states


class SimpleDraftModel(nn.Module):
    """简单的草稿模型 - 只使用基础模型的前3层，层间插入交叉注意力"""
    
    def __init__(self, config: Dict[str, Any], target_model: nn.Module):
        super().__init__()
        self.config = config
        self.target_model = target_model
        
        # 模型配置
        base_config = config['base_model']
        self.hidden_size = base_config['hidden_size']
        self.num_heads = base_config['num_attention_heads']
        
        # 使用前3层（层0, 1, 2）
        self.num_layers = 3
        print(f"使用基础模型的前 {self.num_layers} 层: [0, 1, 2]")
        
        # 提取前3层
        self.layers = nn.ModuleList([
            target_model.model.layers[i] for i in range(self.num_layers)
        ])
        
        # 在每层之间插入交叉注意力（在Layer 0和Layer 1之间，Layer 1和Layer 2之间）
        self.cross_attentions = nn.ModuleList([
            SimpleCrossAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads
            )
            for _ in range(self.num_layers - 1)  # 3层需要2个交叉注意力
        ])
        
        # 复制其他组件
        self.embed_tokens = target_model.model.embed_tokens
        self.norm = target_model.model.norm
        self.lm_head = target_model.lm_head
        
        # 参数统计
        self._print_parameter_count()
    
    def _print_parameter_count(self):
        """打印参数统计"""
        draft_params = sum(p.numel() for p in self.parameters())
        target_params = sum(p.numel() for p in self.target_model.parameters())
        
        # 计算使用的层参数
        used_layer_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.layers
        )
        cross_attn_params = sum(
            sum(p.numel() for p in ca.parameters())
            for ca in self.cross_attentions
        )
        
        print(f"\n=== 简单草稿模型参数统计 ===")
        print(f"使用的层数: {self.num_layers} / {len(self.target_model.model.layers)}")
        print(f"使用的层参数: {used_layer_params:,}")
        print(f"交叉注意力参数: {cross_attn_params:,}")
        print(f"草稿模型总参数: {draft_params:,}")
        print(f"目标模型总参数: {target_params:,}")
        print(f"参数压缩比: {draft_params/target_params:.2%}")
        print("="*50)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            **kwargs: 其他参数（如position_ids等）
        
        Returns:
            dict包含logits和hidden_states
        """
        # 嵌入层
        hidden_states = self.embed_tokens(input_ids)
        
        # 计算position_ids（如果需要）
        if 'position_ids' not in kwargs:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = torch.arange(
                seq_len, 
                device=hidden_states.device, 
                dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
            kwargs['position_ids'] = position_ids
        
        # 计算position_embeddings（如果需要）
        if 'position_embeddings' not in kwargs and hasattr(self.target_model, 'model') and hasattr(self.target_model.model, 'rotary_emb'):
            position_embeddings = self.target_model.model.rotary_emb(
                hidden_states, 
                kwargs['position_ids']
            )
            kwargs['position_embeddings'] = position_embeddings
        
        # 修复attention_mask的dtype
        if attention_mask is not None and attention_mask.dtype == torch.long:
            attention_mask = None  # 让模型使用默认的causal mask
        
        # 逐层处理，在层之间插入交叉注意力
        for i, layer in enumerate(self.layers):
            # 执行当前层
            layer_output = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # 提取hidden_states
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
            
            # 如果不是最后一层，在层之间插入交叉注意力
            if i < len(self.layers) - 1:
                # 使用当前层的输出作为query和context（自注意力形式）
                # 这样可以增强层内的信息流动
                hidden_states = self.cross_attentions[i](
                    hidden_states=hidden_states,
                    context=None,  # None表示使用hidden_states作为context（自注意力）
                    attention_mask=attention_mask
                )
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 输出头
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 **kwargs) -> torch.Tensor:
        """生成函数"""
        current_input = input_ids
        generated = []
        
        eos_token_id = None
        if hasattr(self.target_model, 'config') and hasattr(self.target_model.config, 'eos_token_id'):
            eos_token_id = self.target_model.config.eos_token_id
        
        for i in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(current_input, **kwargs)
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 采样下一个token（贪心解码）
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_token)
            
            # 更新输入
            current_input = torch.cat([current_input, next_token], dim=1)
            
            # 停止条件
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return torch.cat(generated, dim=1) if generated else input_ids

