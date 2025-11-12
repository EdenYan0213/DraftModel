import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import numpy as np

class SpeculativeDecoder:
    """推测解码器 - 使用草稿模型和目标模型进行加速推理"""
    
    def __init__(self, draft_model, target_model, tokenizer, gamma: int = 4):
        """
        Args:
            draft_model: 草稿模型
            target_model: 目标模型
            tokenizer: tokenizer
            gamma: 草稿模型生成的token数量
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.device = next(draft_model.parameters()).device
        
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 **kwargs) -> torch.Tensor:
        """
        推测解码生成
        
        Args:
            input_ids: 输入token IDs
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
        """
        self.draft_model.eval()
        self.target_model.eval()
        
        generated_tokens = []
        current_input = input_ids.clone()
        
        for _ in range(max_new_tokens // self.gamma + 1):
            # 1. 草稿模型生成gamma个token
            draft_tokens = self._draft_generate(
                current_input, 
                num_tokens=self.gamma,
                temperature=temperature,
                top_p=top_p
            )
            
            # 2. 目标模型验证并接受
            accepted_tokens = self._verify_and_accept(
                current_input,
                draft_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # 3. 更新输入
            if len(accepted_tokens) > 0:
                generated_tokens.extend(accepted_tokens)
                current_input = torch.cat([
                    current_input,
                    torch.tensor([accepted_tokens], device=self.device)
                ], dim=1)
            else:
                # 如果没有接受任何token，使用目标模型生成一个
                next_token = self._target_generate_one(
                    current_input,
                    temperature=temperature,
                    top_p=top_p
                )
                generated_tokens.append(next_token.item())
                current_input = torch.cat([
                    current_input,
                    next_token.unsqueeze(0)
                ], dim=1)
            
            # 检查停止条件
            if len(generated_tokens) >= max_new_tokens:
                break
            
            if generated_tokens and generated_tokens[-1] == self.tokenizer.eos_token_id:
                break
        
        return torch.tensor([generated_tokens[:max_new_tokens]], device=self.device)
    
    def _draft_generate(self, 
                       input_ids: torch.Tensor,
                       num_tokens: int,
                       temperature: float = 1.0,
                       top_p: float = 0.9,
                       query_text: Optional[str] = None) -> List[int]:
        """
        草稿模型并行生成多个token
        
        优化：使用KV cache加速，但仍然是逐个生成（因为需要前一个token的结果）
        真正的并行需要更复杂的实现，这里先优化为使用use_cache
        """
        tokens = []
        current = input_ids.clone()
        past_key_values = None  # 用于KV cache
        
        for _ in range(num_tokens):
            # 使用past_key_values加速（如果模型支持）
            outputs = self.draft_model(
                input_ids=current,
                retrieve_knowledge=True,
                query_text=query_text,
                use_cache=True if hasattr(self.draft_model, 'forward') else False
            )
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens.append(next_token.item())
            
            current = torch.cat([current, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return tokens
    
    def _verify_and_accept(self,
                           input_ids: torch.Tensor,
                           draft_tokens: List[int],
                           temperature: float = 1.0,
                           top_p: float = 0.9) -> List[int]:
        """
        并行验证并接受草稿token
        
        关键优化：
        1. 一次性构建包含所有draft tokens的完整序列
        2. 目标模型一次性前向传播（利用KV cache，只需一次前向传播）
        3. 并行获取所有位置的logits
        4. 按顺序验证（因为后续token依赖前面的token）
        
        这样目标模型只需1次前向传播，而不是len(draft_tokens)次
        """
        if not draft_tokens:
            return []
        
        # 构建包含所有draft tokens的完整序列
        # 形状: (batch, input_len + num_draft_tokens)
        draft_sequence = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=self.device)
        ], dim=1)
        
        # 创建attention mask
        attention_mask = torch.ones_like(draft_sequence, dtype=torch.long)
        
        # 目标模型一次性前向传播（并行处理所有token位置）
        # 关键：只需一次前向传播，而不是len(draft_tokens)次
        outputs = self.target_model(
            input_ids=draft_sequence,
            attention_mask=attention_mask
        )
        
        # 获取所有位置的logits
        # logits形状: (batch, seq_len, vocab_size)
        # seq_len = input_len + num_draft_tokens
        all_logits = outputs.logits / temperature
        
        accepted = []
        input_len = input_ids.shape[1]
        
        # 按顺序验证每个draft token（虽然logits是并行计算的）
        # 关键理解：
        # - logits[i] 是在序列位置i后预测下一个token的概率分布
        # - 对于draft token t_i，我们需要验证：在input + t_0 + ... + t_{i-1}后，预测t_i的概率
        # - 序列: [input(0..n-1), t0(n), t1(n+1), t2(n+2), t3(n+3), t4(n+4)]
        # - logits[n-1] 对应在input后预测t0的概率
        # - logits[n] 对应在input+t0后预测t1的概率
        # - 所以对于draft token i，应该看logits[input_len + i - 1]
        
        for i, draft_token in enumerate(draft_tokens):
            # 计算logits索引
            # 对于draft token i，我们需要验证在位置(input_len + i - 1)后预测它的概率
            pos_idx = input_len + i - 1
            if pos_idx < 0:
                pos_idx = 0
            elif pos_idx >= all_logits.shape[1]:
                pos_idx = all_logits.shape[1] - 1
            
            # 获取该位置的logits（并行计算得到的）
            target_logits = all_logits[:, pos_idx, :]
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(target_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                target_logits[indices_to_remove] = float('-inf')
            
            target_probs = F.softmax(target_logits, dim=-1)
            
            # 计算接受概率
            draft_token_id = draft_token
            accept_prob = target_probs[0, draft_token_id].item()
            
            # 接受或拒绝
            # 使用确定性策略：如果token相同或接受概率>阈值则接受
            target_token_id = torch.argmax(target_logits, dim=-1).item()
            if draft_token_id == target_token_id or accept_prob > 0.1:
                accepted.append(draft_token)
            else:
                # 拒绝，从目标模型分布中采样
                target_token = torch.multinomial(target_probs, num_samples=1)
                accepted.append(target_token.item())
                break  # 一旦拒绝，停止接受后续token
        
        return accepted
    
    def _target_generate_one(self,
                            input_ids: torch.Tensor,
                            temperature: float = 1.0,
                            top_p: float = 0.9) -> torch.Tensor:
        """目标模型生成一个token"""
        outputs = self.target_model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        
        # Top-p采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

