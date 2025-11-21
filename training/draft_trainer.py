import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys
try:
    from .attention_alignment import compute_alignment_loss, AttentionAlignmentModule
    ATTENTION_ALIGNMENT_AVAILABLE = True
except ImportError:
    ATTENTION_ALIGNMENT_AVAILABLE = False
    print("⚠ 注意力对齐模块不可用，将跳过注意力对齐损失")
import time
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
        self.learning_rate = float(training_config['learning_rate'])  # 确保是float
        self.num_epochs = int(training_config['num_epochs'])
        self.max_seq_length = int(training_config['max_seq_length'])
        self.grad_accum_steps = int(training_config['gradient_accumulation_steps'])
        self.use_distillation = training_config.get('use_knowledge_distillation', True)
        self.kl_weight = float(training_config.get('kl_divergence_weight', 0.8))
        self.acceptance_weight = float(training_config.get('acceptance_loss_weight', 0.3))  # 接受概率损失权重
        self.alignment_weight = float(training_config.get('alignment_loss_weight', 0.2))  # 注意力对齐损失权重
        self.newline_token_weight = float(training_config.get('newline_token_weight', 0.3))  # 换行符token权重（降低）
        
        # 识别换行符token ID
        self.newline_token_ids = set()
        for test_str in ['\n', '\r\n', '\r']:
            try:
                token_ids = self.tokenizer.encode(test_str, add_special_tokens=False)
                self.newline_token_ids.update(token_ids)
            except:
                pass
        if self.newline_token_ids:
            print(f"识别到换行符token IDs: {self.newline_token_ids}，训练时将应用权重 {self.newline_token_weight}")
        
        # 注意力对齐模块
        if ATTENTION_ALIGNMENT_AVAILABLE and self.alignment_weight > 0:
            self.alignment_module = AttentionAlignmentModule(alpha=0.3, loss_type='kl')
            print(f"✓ 启用注意力对齐损失，权重: {self.alignment_weight}")
        else:
            self.alignment_module = None
        
        # 训练状态
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_loss = float('inf')
        
    def setup_training(self, total_steps: int):
        """设置训练组件"""
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.draft_model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
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
            # 准备输入
            inputs = self._prepare_batch(batch)
            texts = batch.get('text', None)  # 获取原始文本用于检索知识
            
            # 前向传播（传入文本用于知识检索）
            loss = self._compute_training_loss(inputs, texts=texts)
            
            # 检查loss是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠ 警告: 检测到NaN/Inf loss，跳过此批次")
                print(f"   Step: {step}, Loss: {loss.item()}")
                continue
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度累积
            if (step + 1) % self.grad_accum_steps == 0:
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(
                    self.draft_model.parameters(),
                    max_norm=1.0
                )
                
                # 检查梯度是否包含NaN
                has_nan_grad = False
                for name, param in self.draft_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"\n⚠ 警告: 参数 {name} 的梯度包含NaN/Inf")
                            has_nan_grad = True
                            break
                
                if not has_nan_grad:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                else:
                    print("⚠ 跳过优化步骤（梯度包含NaN/Inf）")
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # 更新进度条
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_epoch_loss = total_loss / total_batches
        return avg_epoch_loss
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """准备训练批次"""
        texts = batch['text']
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        )
        
        # 确保attention_mask存在
        # 如果tokenizer没有生成attention_mask（pad_token == eos_token的情况）
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            
            # 创建attention_mask：非pad token的位置为1
            attention_mask = (inputs['input_ids'] != pad_token_id).long()
            inputs['attention_mask'] = attention_mask
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _compute_training_loss(self, inputs: Dict[str, torch.Tensor], texts: Optional[List[str]] = None) -> torch.Tensor:
        """计算训练损失"""
        # 目标模型输出（教师），需要attention权重
        with torch.no_grad():
            target_outputs = self.target_model(**inputs, output_attentions=(self.alignment_module is not None))
            target_logits = target_outputs.logits
            target_attentions = getattr(target_outputs, 'attentions', None)
        
        # 检索知识缓存（如果有）
        knowledge_cache = None
        if self.knowledge_cache_manager is not None and texts is not None:
            # 为每个样本检索相关的KV缓存
            batch_kv_caches = []
            for text in texts:
                # 尝试检索KV缓存
                kv_result = self.knowledge_cache_manager.retrieve(text)
                if kv_result is not None:
                    knowledge_keys, knowledge_values = kv_result
                    # 移动到正确的设备
                    knowledge_keys = knowledge_keys.to(self.device)
                    knowledge_values = knowledge_values.to(self.device)
                    batch_kv_caches.append((knowledge_keys, knowledge_values))
                else:
                    batch_kv_caches.append(None)
            
            # 如果批次中所有样本都有缓存，使用第一个（简化处理）
            # 实际可以更智能地处理批次中不同的缓存
            if batch_kv_caches[0] is not None:
                knowledge_cache = batch_kv_caches[0]
        
        # 草稿模型输出（学生，使用知识缓存），需要attention权重
        draft_outputs = self.draft_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            knowledge_cache=knowledge_cache,  # 传递KV缓存
            retrieve_knowledge=(knowledge_cache is None),  # 如果没有提供缓存，尝试检索
            query_text=texts[0] if texts else None,
            output_attentions=(self.alignment_module is not None)  # 需要attention
        )
        draft_logits = draft_outputs['logits']
        draft_attentions = draft_outputs.get('attentions')
        
        # 检查logits是否包含NaN或Inf
        if torch.isnan(draft_logits).any() or torch.isinf(draft_logits).any():
            print(f"\n⚠ 警告: 草稿模型logits包含NaN/Inf")
            print(f"   NaN数量: {torch.isnan(draft_logits).sum().item()}")
            print(f"   Inf数量: {torch.isinf(draft_logits).sum().item()}")
            # 返回一个较大的loss，但不会导致NaN
            return torch.tensor(10.0, device=self.device, requires_grad=True)
        
        # ========== 改进的接受概率损失计算 ==========
        # 使用贪心对齐：直接最大化目标模型对草稿模型贪心预测的概率
        acceptance_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if self.acceptance_weight > 0:
            try:
                # 方法1: 贪心对齐（更直接，更符合推理时的接受逻辑）
                # 草稿模型的贪心预测（argmax）
                draft_predictions = torch.argmax(draft_logits[:, :-1, :], dim=-1)  # (batch, seq-1)
                
                # 目标模型的概率分布
                target_probs = F.softmax(target_logits[:, :-1, :].detach(), dim=-1)  # (batch, seq-1, vocab)
                
                # 获取目标模型对草稿模型预测token的概率
                target_probs_for_draft = torch.gather(
                    target_probs,
                    dim=-1,
                    index=draft_predictions.unsqueeze(-1)
                ).squeeze(-1)  # (batch, seq-1)
                
                # 贪心对齐损失：使用平方损失，让小概率的惩罚更大
                # 这样当目标模型对草稿预测的概率很小时，损失会更大，迫使模型更好地对齐
                greedy_loss = -torch.mean((torch.log(target_probs_for_draft + 1e-8)) ** 2)
                
                # 方法2: 期望对齐（作为辅助，保持原有的平滑性）
                draft_probs = F.softmax(draft_logits[:, :-1, :], dim=-1)  # (batch, seq-1, vocab)
                acceptance_probs = torch.sum(draft_probs * target_probs, dim=-1)  # (batch, seq-1)
                expected_loss = -torch.mean(torch.log(acceptance_probs + 1e-8))
                
                # 组合两种损失（贪心对齐为主，期望对齐为辅）
                acceptance_loss = 0.7 * greedy_loss + 0.3 * expected_loss
                
                # 检查acceptance_loss是否为NaN
                if torch.isnan(acceptance_loss) or torch.isinf(acceptance_loss):
                    print(f"\n⚠ 警告: 接受概率损失为NaN/Inf，跳过")
                    acceptance_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            except Exception as e:
                print(f"\n⚠ 警告: 计算接受概率损失时出错: {e}，跳过")
                import traceback
                traceback.print_exc()
                acceptance_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ========== 注意力引导的对齐损失 ==========
        alignment_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if self.alignment_module is not None and draft_attentions is not None and target_attentions is not None:
            try:
                # 提取最后一层的attention
                draft_attn = draft_attentions[-1] if isinstance(draft_attentions, (list, tuple)) else draft_attentions
                target_attn = target_attentions[-1] if isinstance(target_attentions, (list, tuple)) else target_attentions
                
                # 计算对齐损失
                alignment_loss = compute_alignment_loss(
                    draft_logits,
                    target_logits,
                    draft_attn,
                    target_attn,
                    loss_type='kl'
                )
                
                if torch.isnan(alignment_loss) or torch.isinf(alignment_loss):
                    print(f"\n⚠ 警告: 注意力对齐损失为NaN/Inf，跳过")
                    alignment_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            except Exception as e:
                print(f"\n⚠ 警告: 计算注意力对齐损失时出错: {e}，跳过")
                alignment_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.use_distillation:
            # KL散度损失（知识蒸馏）
            try:
                # 确保logits是数值稳定的
                draft_logits_shifted = draft_logits[:, :-1, :]
                target_logits_shifted = target_logits[:, :-1, :].detach()
                
                # 检查target_logits是否包含NaN
                if torch.isnan(target_logits_shifted).any() or torch.isinf(target_logits_shifted).any():
                    print(f"\n⚠ 警告: 目标模型logits包含NaN/Inf，仅使用CE loss")
                    kl_loss = torch.tensor(0.0, device=self.device)
                else:
                    kl_loss = F.kl_div(
                        F.log_softmax(draft_logits_shifted, dim=-1),
                        F.softmax(target_logits_shifted, dim=-1),
                        reduction='batchmean'
                    )
                    
                    # 检查KL loss是否为NaN
                    if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                        print(f"\n⚠ 警告: KL loss为NaN/Inf，仅使用CE loss")
                        kl_loss = torch.tensor(0.0, device=self.device)
            except Exception as e:
                print(f"\n⚠ 警告: 计算KL loss时出错: {e}，仅使用CE loss")
                kl_loss = torch.tensor(0.0, device=self.device)
            
            # 交叉熵损失（对换行符应用权重惩罚）
            # 创建权重张量
            target_ids = inputs['input_ids'][:, 1:].reshape(-1)  # (batch * seq)
            weights = torch.ones_like(target_ids, dtype=torch.float32, device=self.device)
            
            # 对换行符token应用降低的权重
            if self.newline_token_ids:
                for nl_id in self.newline_token_ids:
                    if nl_id < draft_logits.size(-1):
                        weights[target_ids == nl_id] = self.newline_token_weight
            
            # 使用加权交叉熵损失
            ce_loss = F.cross_entropy(
                draft_logits[:, :-1, :].reshape(-1, draft_logits.size(-1)),
                target_ids,
                weight=None,  # 不使用全局权重，而是手动应用
                ignore_index=self.tokenizer.pad_token_id,
                reduction='none'  # 先不reduce，手动应用权重
            )
            # 应用权重
            ce_loss = (ce_loss * weights).mean()
            
            # 检查CE loss是否为NaN
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print(f"\n⚠ 警告: CE loss为NaN/Inf")
                ce_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
            
            # 组合损失（添加接受概率损失和注意力对齐损失）
            # 修复：只对KL和CE损失进行归一化，接受损失和对齐损失直接加上（不归一化）
            # 这样接受损失的实际权重就是配置的值（0.7），而不是被归一化到0.35
            kl_ce_total = self.kl_weight + (1 - self.kl_weight)  # 应该是1.0
            kl_w = self.kl_weight / kl_ce_total if kl_ce_total > 0 else 0.5
            ce_w = (1 - self.kl_weight) / kl_ce_total if kl_ce_total > 0 else 0.5
            
            # 接受损失和对齐损失不归一化，直接使用配置的权重
            acc_w = self.acceptance_weight
            align_w = self.alignment_weight
            
            total_loss = kl_w * kl_loss + ce_w * ce_loss + acc_w * acceptance_loss + align_w * alignment_loss
        else:
            # 仅使用交叉熵损失（对换行符应用权重惩罚）
            target_ids = inputs['input_ids'][:, 1:].reshape(-1)
            weights = torch.ones_like(target_ids, dtype=torch.float32, device=self.device)
            
            # 对换行符token应用降低的权重
            if self.newline_token_ids:
                for nl_id in self.newline_token_ids:
                    if nl_id < draft_logits.size(-1):
                        weights[target_ids == nl_id] = self.newline_token_weight
            
            total_loss = F.cross_entropy(
                draft_logits[:, :-1, :].reshape(-1, draft_logits.size(-1)),
                target_ids,
                ignore_index=self.tokenizer.pad_token_id,
                reduction='none'
            )
            # 应用权重
            total_loss = (total_loss * weights).mean()
            
            # 检查loss是否为NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n⚠ 警告: CE loss为NaN/Inf")
                total_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n⚠ 警告: 总loss为NaN/Inf，使用默认值")
            total_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
        
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
                
                # 确保attention_mask存在且正确
                if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
                    # 创建attention_mask：非pad token的位置为1
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = self.tokenizer.eos_token_id
                    
                    attention_mask = (inputs['input_ids'] != pad_token_id).long()
                    inputs['attention_mask'] = attention_mask
                
                loss = self._compute_training_loss(inputs, texts=texts)
                total_loss += loss.item()
        
        avg_loss = total_loss / total_batches
        
        metrics = {
            'val_loss': avg_loss,
            'perplexity': math.exp(avg_loss) if avg_loss < 10 else float('inf')  # 防止溢出
        }
        
        return metrics
    
    def evaluate_acceptance_rate(self, test_prompts: List[str], max_new_tokens: int = 5) -> float:
        """评估接受率"""
        self.draft_model.eval()
        self.target_model.eval()
        
        acceptance_rates = []
        
        with torch.no_grad():
            for prompt in test_prompts:
                # 编码输入
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 草稿模型生成
                draft_tokens = self.draft_model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_new_tokens,
                    retrieve_knowledge=True
                )
                
                # 目标模型生成（用于验证）
                target_outputs = self.target_model.generate(
                    inputs['input_ids'],
                    max_new_tokens=draft_tokens.size(1),
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # 计算接受率
                total_tokens = draft_tokens.size(1)
                accepted = 0
                
                for i in range(total_tokens):
                    draft_token = draft_tokens[0, i] if draft_tokens.dim() > 1 else draft_tokens[i]
                    target_token = target_outputs.sequences[0, inputs['input_ids'].size(1) + i]
                    
                    if draft_token == target_token:
                        accepted += 1
                
                acceptance_rate = accepted / total_tokens if total_tokens > 0 else 0
                acceptance_rates.append(acceptance_rate)
                
                print(f"提示: {prompt[:50]}...")
                print(f"接受率: {acceptance_rate:.3f}")
        
        avg_acceptance = sum(acceptance_rates) / len(acceptance_rates)
        print(f"平均接受率: {avg_acceptance:.4f}")
        
        return avg_acceptance
    
    def save_checkpoint(self, epoch: int, loss: float, filepath: str):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.draft_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'sampled_indices': self.draft_model.sampled_indices
        }
        
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath} (loss: {loss:.4f})")
    
    def load_checkpoint(self, filepath: str):
        """加载训练检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.draft_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        
        print(f"检查点已加载: {filepath}")
        print(f"恢复训练: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
        
        return checkpoint['epoch'], checkpoint['loss']

