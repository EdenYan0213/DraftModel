#!/usr/bin/env python3
"""
训练知识增强的草稿模型
"""

import os
import sys
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.draft_model import Qwen3DraftModel
from models.knowledge_cache import KnowledgeCacheManager
from models.utils import get_device, print_device_info
from training.draft_trainer import DraftModelTrainer
from training.data_utils import create_sample_dataloader

def main():
    """主函数 - 训练草稿模型（步骤2）"""
    print("="*70)
    print("Qwen3-0.6B 草稿模型训练工具 (知识增强版)")
    print("="*70)
    print("注意: 这是训练步骤（步骤2），请确保已先构建知识缓存（步骤1）")
    print("="*70)
    
    config_path = "configs/qwen3_0.6b_config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    knowledge_enabled = config.get('knowledge_enhancement', {}).get('enabled', False)
    print(f"\n知识增强: {'✓ 已启用' if knowledge_enabled else '✗ 未启用'}")
    
    loader = Qwen3Loader(config_path)
    
    print("\n1. 加载基础模型...")
    target_model = loader.load_target_model()
    tokenizer = loader.load_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n2. 加载知识缓存...")
    cache_path = "output/knowledge_cache/knowledge_cache.pth"
    knowledge_cache_manager = None
    
    if os.path.exists(cache_path):
        print(f"从 {cache_path} 加载知识缓存...")
        base_config = config['base_model']
        knowledge_config = config.get('knowledge_enhancement', {})
        
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=base_config['hidden_size'],
            use_vector_retrieval=knowledge_config.get('use_vector_retrieval', True),
            embedding_model_name=knowledge_config.get('embedding_model_name', None),
            target_model=target_model,
            tokenizer=tokenizer
        )
        knowledge_cache_manager.load(cache_path)
        print(f"✓ 知识缓存加载完成，共 {len(knowledge_cache_manager.knowledge_cache)} 个知识项")
        
        # 检查是否有qa_pairs（用于训练数据）
        if hasattr(knowledge_cache_manager, 'qa_pairs') and len(knowledge_cache_manager.qa_pairs) > 0:
            print(f"✓ 问答对数量: {len(knowledge_cache_manager.qa_pairs)} (将用作训练数据)")
        else:
            print("⚠ 警告: 知识缓存中没有问答对文本（qa_pairs）")
            print("  训练数据可能不完整，建议重新构建知识缓存：")
            print("  python scripts/generate_knowledge_from_model.py")
    else:
        print(f"⚠ 知识缓存文件不存在: {cache_path}")
        if knowledge_enabled:
            print("\n" + "="*70)
            print("错误: 知识缓存不存在")
            print("="*70)
            print("请先构建知识缓存（步骤1）：")
            print("  python scripts/generate_knowledge_from_model.py")
            print("\n或者使用旧版构建脚本：")
            print("  python scripts/build_knowledge_cache.py")
            print("\n详细说明请参考: USAGE.md")
            print("="*70)
            return
        print("继续训练，但不使用知识增强...")
    
    print(f"\n3. 创建草稿模型...")
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
    
    device = get_device('cpu')
    print_device_info(device)
    
    draft_model = draft_model.to(device)
    target_model = target_model.to(device)
    
    print("\n4. 准备训练数据...")
    train_dataloader, val_dataloader = create_sample_dataloader(
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['training']['max_seq_length'],
        knowledge_cache_manager=knowledge_cache_manager  # 传递知识缓存管理器
    )
    
    print("\n5. 初始化训练器...")
    trainer = DraftModelTrainer(
        config=config,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        knowledge_cache_manager=knowledge_cache_manager
    )
    
    total_steps = len(train_dataloader) * config['training']['num_epochs']
    trainer.setup_training(total_steps)
    
    print("\n6. 开始训练...")
    print("="*70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    checkpoint_dir = "output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 清理旧的检查点文件
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pth'):
            old_path = os.path.join(checkpoint_dir, f)
            try:
                os.remove(old_path)
                print(f"清理旧检查点: {f}")
            except:
                pass
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\n{'='*70}", flush=True)
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}", flush=True)
        print(f"{'='*70}", flush=True)
        
        train_loss = trainer.train_epoch(train_dataloader, epoch)
        print(f"\n训练损失: {train_loss:.4f}", flush=True)
        
        val_metrics = trainer.validate(val_dataloader)
        val_loss = val_metrics['val_loss']
        print(f"验证损失: {val_loss:.4f}")
        if 'perplexity' in val_metrics:
            print(f"验证困惑度: {val_metrics['perplexity']:.2f}")
        
        # 保存最佳模型（如果更好）
        if val_loss < best_val_loss:
            # 删除旧的最佳模型
            if best_epoch > 0:
                old_best_path = os.path.join(checkpoint_dir, f"best_draft_model_epoch{best_epoch}.pth")
                if os.path.exists(old_best_path):
                    os.remove(old_best_path)
            
            best_val_loss = val_loss
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(checkpoint_dir, f"best_draft_model_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': draft_model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'sampled_indices': draft_model.sampled_indices,
                'knowledge_enabled': knowledge_enabled
            }, checkpoint_path)
            print(f"✓ 最佳模型已保存: {checkpoint_path} (loss: {val_loss:.4f})")
        
        # 每5轮保存一次模型（如果是最佳模型，则不重复保存）
        if (epoch + 1) % 5 == 0:
            # 如果当前epoch就是最佳模型，已经保存过了，跳过
            if epoch + 1 != best_epoch:
                epoch_checkpoint = os.path.join(checkpoint_dir, f"draft_model_epoch{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': draft_model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'sampled_indices': draft_model.sampled_indices,
                    'knowledge_enabled': knowledge_enabled
                }, epoch_checkpoint)
                print(f"✓ Epoch {epoch+1} 检查点已保存: {epoch_checkpoint}")
            else:
                print(f"✓ Epoch {epoch+1} 已作为最佳模型保存，跳过重复保存")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"知识增强: {'✓ 已启用' if knowledge_enabled else '✗ 未启用'}")
    print(f"\n模型保存位置: {checkpoint_dir}")
    
    # 列出最终保存的模型
    saved_models = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pth'):
            saved_models.append(f)
    saved_models.sort()
    print(f"\n最终保存的模型文件 ({len(saved_models)} 个):")
    for model_file in saved_models:
        print(f"  - {model_file}")

if __name__ == "__main__":
    main()

