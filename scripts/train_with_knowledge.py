#!/usr/bin/env python3
"""
使用知识增强训练草稿模型
"""

import os
import sys
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_enhanced_draft import Qwen3DraftModel
from models.knowledge_cache import KnowledgeCacheManager
from training.draft_trainer import DraftModelTrainer
from training.data_utils import create_sample_dataloader

def main():
    """主函数"""
    print("="*70)
    print("Qwen3-0.6B 草稿模型训练工具 (知识增强版)")
    print("="*70)
    
    # 加载配置
    config_path = "/Users/chuang.yan/PycharmProjects/CrossAndAttention/configs/qwen3_0.6b_config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查知识增强是否启用
    knowledge_enabled = config.get('knowledge_enhancement', {}).get('enabled', False)
    print(f"\n知识增强: {'✓ 已启用' if knowledge_enabled else '✗ 未启用'}")
    
    # 初始化加载器
    loader = Qwen3Loader(config_path)
    
    # 加载目标模型和tokenizer
    print("\n1. 加载基础模型...")
    target_model = loader.load_target_model()
    tokenizer = loader.load_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载知识缓存
    print("\n2. 加载知识缓存...")
    cache_path = "output/knowledge_cache/knowledge_cache.pth"
    knowledge_cache_manager = None
    
    if os.path.exists(cache_path):
        print(f"从 {cache_path} 加载知识缓存...")
        # 使用weights_only=False以支持numpy数组（embeddings）
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        
        base_config = config['base_model']
        knowledge_config = config.get('knowledge_enhancement', {})
        
        # 创建知识缓存管理器（启用向量检索）
        use_vector_retrieval = knowledge_config.get('use_vector_retrieval', True)
        embedding_model_name = knowledge_config.get('embedding_model_name', None)
        
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=base_config['hidden_size'],
            num_heads=base_config['num_attention_heads'],
            cache_dim=knowledge_config.get('cache_dim', 512),
            use_vector_retrieval=use_vector_retrieval,
            embedding_model_name=embedding_model_name
        )
        
        # 恢复缓存数据
        knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})
        knowledge_cache_manager.compressed_cache = cache_data.get('compressed_cache', {})
        
        # 恢复embeddings（如果存在）
        if 'knowledge_embeddings' in cache_data:
            knowledge_cache_manager.knowledge_embeddings = cache_data['knowledge_embeddings']
            print(f"✓ 已恢复 {len(knowledge_cache_manager.knowledge_embeddings)} 个知识项的embeddings")
        
        # 恢复压缩/解压缩器（如果保存了）
        if 'compressor_state' in cache_data:
            knowledge_cache_manager.kv_compressor.load_state_dict(cache_data['compressor_state'])
        if 'decompressor_state' in cache_data:
            knowledge_cache_manager.kv_decompressor.load_state_dict(cache_data['decompressor_state'])
        
        print(f"✓ 知识缓存加载完成，共 {len(knowledge_cache_manager.kv_cache)} 个知识项")
        print(f"  检索方法: {knowledge_cache_manager.get_retrieval_method()}")
    else:
        print(f"⚠ 知识缓存文件不存在: {cache_path}")
        if knowledge_enabled:
            print("请先运行: python scripts/build_knowledge_cache.py")
            return
        print("继续训练，但不使用知识增强...")
    
    # 创建草稿模型（传入知识缓存管理器）
    print(f"\n3. 创建草稿模型...")
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
    
    # 移动到设备（优先CUDA，其次CPU，MPS有兼容性问题暂时禁用）
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用设备: CUDA")
    else:
        print(f"使用设备: CPU (MPS有兼容性问题，使用CPU)")
    
    draft_model = draft_model.to(device)
    target_model = target_model.to(device)
    
    # 创建数据加载器
    print("\n4. 准备训练数据...")
    train_dataloader = create_sample_dataloader(
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['training']['max_seq_length'],
        num_samples=1000  # 减少样本数以加快训练
    )
    
    val_dataloader = create_sample_dataloader(
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['training']['max_seq_length'],
        num_samples=200
    )
    
    # 创建训练器
    print("\n5. 初始化训练器...")
    trainer = DraftModelTrainer(
        config=config,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        knowledge_cache_manager=knowledge_cache_manager
    )
    
    # 设置训练
    total_steps = len(train_dataloader) * config['training']['num_epochs']
    trainer.setup_training(total_steps)
    
    # 训练
    print("\n6. 开始训练...")
    print("="*70)
    
    best_val_loss = float('inf')
    checkpoint_dir = "output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\n{'='*70}", flush=True)
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # 训练一个epoch
        train_loss = trainer.train_epoch(train_dataloader, epoch)
        print(f"\n训练损失: {train_loss:.4f}", flush=True)
        
        # 验证
        val_metrics = trainer.validate(val_dataloader)
        val_loss = val_metrics['val_loss']
        print(f"验证损失: {val_loss:.4f}")
        if 'perplexity' in val_metrics:
            print(f"验证困惑度: {val_metrics['perplexity']:.2f}")
        
        # 保存最佳模型（如果更好）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_draft_model_knowledge_epoch{epoch+1}.pth")
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
        
        # 每5个epoch保存一次模型权重
        if (epoch + 1) % 5 == 0:
            epoch_checkpoint = os.path.join(checkpoint_dir, f"draft_model_knowledge_epoch{epoch+1}.pth")
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
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final_draft_model_knowledge.pth")
    torch.save({
        'epoch': config['training']['num_epochs'],
        'model_state_dict': draft_model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'sampled_indices': draft_model.sampled_indices,
        'knowledge_enabled': knowledge_enabled
    }, final_path)
    print(f"\n✓ 最终模型已保存: {final_path}")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"知识增强: {'✓ 已启用' if knowledge_enabled else '✗ 未启用'}")
    print(f"\n模型保存位置: {checkpoint_dir}")

if __name__ == "__main__":
    main()

