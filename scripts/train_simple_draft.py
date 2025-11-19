#!/usr/bin/env python3
"""
训练简单草稿模型 - 只使用前3层，层间插入交叉注意力
"""

import os
import sys
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.simple_draft_model import SimpleDraftModel
from training.draft_trainer import DraftModelTrainer
from training.data_utils import create_sample_dataloader

def main():
    """主函数"""
    print("="*70)
    print("简单草稿模型训练工具")
    print("="*70)
    
    # 加载配置
    config_path = "configs/qwen3_0.6b_config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化加载器
    loader = Qwen3Loader(config_path)
    
    # 加载目标模型和tokenizer
    print("\n1. 加载基础模型...")
    target_model = loader.load_target_model(device='cpu')
    tokenizer = loader.load_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 基础模型加载完成，总层数: {len(target_model.model.layers)}")
    
    # 创建简单草稿模型
    print("\n2. 创建简单草稿模型...")
    simple_draft = SimpleDraftModel(config, target_model)
    simple_draft = simple_draft.cpu()
    simple_draft.train()
    target_model.eval()  # 目标模型不训练
    
    print("✓ 简单草稿模型创建完成")
    
    # 创建数据加载器
    print("\n3. 创建数据加载器...")
    training_config = config['training']
    batch_size = int(training_config['batch_size'])
    max_length = int(training_config['max_seq_length'])
    
    train_dataloader = create_sample_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=1000,  # 使用1000个样本
        max_length=max_length
    )
    
    val_dataloader = create_sample_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=200,  # 验证集200个样本
        max_length=max_length
    )
    
    print(f"✓ 训练集: {len(train_dataloader)} 批次")
    print(f"✓ 验证集: {len(val_dataloader)} 批次")
    
    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = DraftModelTrainer(
        config=config,
        draft_model=simple_draft,
        target_model=target_model,
        tokenizer=tokenizer,
        knowledge_cache_manager=None  # 简单模型不使用知识缓存
    )
    
    # 设置训练
    num_epochs = int(training_config['num_epochs'])
    total_steps = len(train_dataloader) * num_epochs
    trainer.setup_training(total_steps)
    
    # 创建输出目录
    output_dir = "output/checkpoints/simple_draft"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n5. 开始训练...")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_loss = trainer.train_epoch(train_dataloader, epoch)
        print(f"\n训练损失: {train_loss:.4f}")
        
        # 验证
        val_results = trainer.validate(val_dataloader)
        val_loss = val_results.get('loss', val_results.get('val_loss', 0.0))
        print(f"验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, f"simple_draft_best_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': simple_draft.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ 保存最佳模型: {checkpoint_path}")
        
        # 每5个epoch保存一次检查点
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f"simple_draft_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': simple_draft.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ 保存检查点: {checkpoint_path}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(output_dir, "simple_draft_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': simple_draft.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, final_checkpoint_path)
    print(f"\n✓ 保存最终模型: {final_checkpoint_path}")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"\n最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存位置: {output_dir}")


if __name__ == "__main__":
    main()

