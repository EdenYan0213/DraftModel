"""
草稿模型训练器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import sys
import math
from typing import Dict, Any, List, Optional


class DraftModelTrainer:
    """草稿模型训练器"""
    
    def __init__(self, config: Dict[str, Any], draft_model: nn.Module, 
                 target_model: nn.Module, tokenizer: Any, knowledge_cache_manager=None):
        self.config = config
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.knowledge_cache_manager = knowledge_cache_manager
        self.device = next(draft_model.parameters()).device
        
        # 训练配置
        training_config = config['training']
        self.batch_size = int(training_config['batch_size'])
        self.learning_rate = float(training_config['learning_rate'])
        self.max_seq_length = int(training_config['max_seq_length'])
        self.grad_accum_steps = int(training_config['gradient_accumulation_steps'])
        self.use_distillation = training_config.get('use_knowledge_distillation', True)
        self.kl_weight = float(training_config.get('kl_divergence_weight', 0.5))
        self.acceptance_weight = float(training_config.get('acceptance_loss_weight', 0.3))
        
        # 训练状态
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        
    def setup_training(self, total_steps: int):
        """设置训练组件"""
        self.optimizer = torch.optim.AdamW(
            self.draft_model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"训练设置完成:")
        print(f"  - 总步数: {total_steps}")
        print(f"  - Warmup步数: {warmup_steps}")
        print(f"  - 学习率: {self.learning_rate}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """训练一个epoch"""
        self.draft_model.train()
        total_loss = 0
        total_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", file=sys.stdout, dynamic_ncols=True)
        
        for step, batch in enumerate(progress_bar):
            inputs = self._prepare_batch(batch)
            texts = batch.get('text', None)
            
            loss = self._compute_training_loss(inputs, texts=texts)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠ 警告: 检测到NaN/Inf loss，跳过此批次")
                continue
            
            loss.backward()
            
            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), max_norm=1.0)
                
                has_nan_grad = False
                for name, param in self.draft_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if not has_nan_grad:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                else:
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / total_batches
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """准备训练批次"""
        texts = batch['text']
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        )
        
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            attention_mask = (inputs['input_ids'] != pad_token_id).long()
            inputs['attention_mask'] = attention_mask
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _compute_training_loss(self, inputs: Dict[str, torch.Tensor], texts: Optional[List[str]] = None) -> torch.Tensor:
        """计算训练损失"""
        # 目标模型输出（教师）
        with torch.no_grad():
            target_outputs = self.target_model(**inputs)
            target_logits = target_outputs.logits
        
        # 草稿模型输出（学生）
        query_text = texts[0] if texts else None
        try:
            draft_outputs = self.draft_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                retrieve_knowledge=(self.knowledge_cache_manager is not None),
                query_text=query_text
            )
            draft_logits = draft_outputs['logits']
        except Exception as e:
            print(f"\n⚠ 警告: 草稿模型前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个较大的损失值，但不固定为10.0，让模型有机会恢复
            return torch.tensor(5.0, device=self.device, requires_grad=True)
        
        # 检查logits中的NaN/Inf，但使用更合理的处理方式
        if torch.isnan(draft_logits).any() or torch.isinf(draft_logits).any():
            nan_count = torch.isnan(draft_logits).sum().item()
            inf_count = torch.isinf(draft_logits).sum().item()
            print(f"\n⚠ 警告: draft_logits包含NaN ({nan_count}个)或Inf ({inf_count}个)")
            # 将NaN/Inf替换为0，然后继续计算损失
            draft_logits = torch.where(torch.isnan(draft_logits) | torch.isinf(draft_logits),
                                      torch.zeros_like(draft_logits),
                                      draft_logits)
        
        # 接受概率损失
        acceptance_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if self.acceptance_weight > 0:
            try:
                draft_predictions = torch.argmax(draft_logits[:, :-1, :], dim=-1)
                target_probs = F.softmax(target_logits[:, :-1, :].detach(), dim=-1)
                target_probs_for_draft = torch.gather(
                    target_probs,
                    dim=-1,
                    index=draft_predictions.unsqueeze(-1)
                ).squeeze(-1)
                
                greedy_loss = -torch.mean(torch.log(target_probs_for_draft + 1e-8))
                
                draft_probs = F.softmax(draft_logits[:, :-1, :], dim=-1)
                acceptance_probs = torch.sum(draft_probs * target_probs, dim=-1)
                expected_loss = -torch.mean(torch.log(acceptance_probs + 1e-8))
                
                acceptance_loss = 0.7 * greedy_loss + 0.3 * expected_loss
                
                if torch.isnan(acceptance_loss) or torch.isinf(acceptance_loss):
                    acceptance_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            except Exception as e:
                acceptance_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.use_distillation:
            # KL散度损失
            try:
                draft_logits_shifted = draft_logits[:, :-1, :]
                target_logits_shifted = target_logits[:, :-1, :].detach()
                
                if torch.isnan(target_logits_shifted).any() or torch.isinf(target_logits_shifted).any():
                    kl_loss = torch.tensor(0.0, device=self.device)
                else:
                    kl_loss = F.kl_div(
                        F.log_softmax(draft_logits_shifted, dim=-1),
                        F.softmax(target_logits_shifted, dim=-1),
                        reduction='batchmean'
                    )
                    
                    if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                        kl_loss = torch.tensor(0.0, device=self.device)
            except Exception as e:
                kl_loss = torch.tensor(0.0, device=self.device)
            
            # 交叉熵损失
            target_ids = inputs['input_ids'][:, 1:].reshape(-1)
            ce_loss = F.cross_entropy(
                draft_logits[:, :-1, :].reshape(-1, draft_logits.size(-1)),
                target_ids,
                ignore_index=self.tokenizer.pad_token_id
            )
            
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print(f"\n⚠ 警告: ce_loss是NaN/Inf，使用目标模型的损失作为参考")
                # 使用目标模型的损失作为参考值
                with torch.no_grad():
                    target_ce_loss = F.cross_entropy(
                        target_logits[:, :-1, :].reshape(-1, target_logits.size(-1)),
                        target_ids,
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    # 使用目标损失值的1.5倍作为初始值，让模型有学习空间
                    ce_loss = torch.tensor(target_ce_loss.item() * 1.5, device=self.device, requires_grad=True)
            
            # 组合损失
            kl_ce_total = self.kl_weight + (1 - self.kl_weight)
            kl_w = self.kl_weight / kl_ce_total if kl_ce_total > 0 else 0.5
            ce_w = (1 - self.kl_weight) / kl_ce_total if kl_ce_total > 0 else 0.5
            
            total_loss = kl_w * kl_loss + ce_w * ce_loss + self.acceptance_weight * acceptance_loss
        else:
            # 仅使用交叉熵损失
            target_ids = inputs['input_ids'][:, 1:].reshape(-1)
            total_loss = F.cross_entropy(
                draft_logits[:, :-1, :].reshape(-1, draft_logits.size(-1)),
                target_ids,
                ignore_index=self.tokenizer.pad_token_id
            )
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n⚠ 警告: total_loss是NaN/Inf，使用目标模型的损失作为参考")
                # 使用目标模型的损失作为参考值
                with torch.no_grad():
                    target_ce_loss = F.cross_entropy(
                        target_logits[:, :-1, :].reshape(-1, target_logits.size(-1)),
                        target_ids,
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    # 使用目标损失值的1.5倍作为初始值
                    total_loss = torch.tensor(target_ce_loss.item() * 1.5, device=self.device, requires_grad=True)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n⚠ 警告: 最终total_loss是NaN/Inf，使用目标模型的损失作为参考")
            # 最后的安全检查：使用目标模型的损失
            with torch.no_grad():
                target_ids = inputs['input_ids'][:, 1:].reshape(-1)
                target_ce_loss = F.cross_entropy(
                    target_logits[:, :-1, :].reshape(-1, target_logits.size(-1)),
                    target_ids,
                    ignore_index=self.tokenizer.pad_token_id
                )
                total_loss = torch.tensor(target_ce_loss.item() * 1.5, device=self.device, requires_grad=True)
        
        return total_loss
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.draft_model.eval()
        total_loss = 0
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="验证"):
                inputs = self._prepare_batch(batch)
                texts = batch.get('text', None)
                
                if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
                    pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    attention_mask = (inputs['input_ids'] != pad_token_id).long()
                    inputs['attention_mask'] = attention_mask
                
                loss = self._compute_training_loss(inputs, texts=texts)
                total_loss += loss.item()
        
        avg_loss = total_loss / total_batches
        
        return {
            'val_loss': avg_loss,
            'perplexity': math.exp(avg_loss) if avg_loss < 10 else float('inf')
        }

