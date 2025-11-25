#!/usr/bin/env python3
"""
推理测试脚本
"""

import os
import sys
import torch
import yaml
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.draft_model import Qwen3DraftModel
from models.knowledge_cache import KnowledgeCacheManager
from models.utils import get_device, print_device_info


def load_model(config_path: str, checkpoint_path: str, device: str = 'auto'):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    loader = Qwen3Loader(config_path)
    target_model = loader.load_target_model(device=device)
    tokenizer = loader.load_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载知识缓存
    cache_path = "output/knowledge_cache/knowledge_cache.pth"
    knowledge_cache_manager = None
    
    if os.path.exists(cache_path):
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=config['base_model']['hidden_size'],
            use_vector_retrieval=True,
            target_model=target_model,
            tokenizer=tokenizer
        )
        knowledge_cache_manager.load(cache_path)
        print(f"✓ 知识缓存已加载，共 {len(knowledge_cache_manager.knowledge_cache)} 个知识项")
    
    # 创建草稿模型
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    draft_model.load_state_dict(checkpoint['model_state_dict'])
    
    draft_model = draft_model.to(device)
    target_model = target_model.to(device)
    draft_model.eval()
    target_model.eval()
    
    print(f"✓ 模型加载完成 (epoch {checkpoint.get('epoch', 'unknown')})")
    return draft_model, target_model, tokenizer, knowledge_cache_manager


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 0.7, top_p: float = 0.9):
    """生成文本"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 检查是否是草稿模型（有knowledge_cache_manager属性）
            if hasattr(model, 'knowledge_cache_manager'):
                # 草稿模型：传入知识检索参数
                outputs = model(
                    input_ids=generated_ids,
                    retrieve_knowledge=True,
                    query_text=prompt
                )
            else:
                # 目标模型：只传入input_ids
                outputs = model(input_ids=generated_ids)
            
            # 处理不同的输出格式
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs.logits
            
            logits = logits[:, -1, :] / temperature
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # 只返回生成的部分（去掉原始prompt）
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # 移除prompt部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def main():
    """主函数"""
    # 设置随机种子确保可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    import numpy as np
    np.random.seed(42)
    
    print("="*70)
    print("推理测试")
    print("="*70)
    
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 查找检查点
    checkpoint_dir = "output/checkpoints"
    checkpoint_files = []
    
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pth') and 'best' in f:
                checkpoint_files.append(os.path.join(checkpoint_dir, f))
    
    if not checkpoint_files:
        final_path = os.path.join(checkpoint_dir, "final_draft_model.pth")
        if os.path.exists(final_path):
            checkpoint_path = final_path
        else:
            print("错误: 未找到检查点文件")
            return
    else:
        checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    
    print(f"使用检查点: {checkpoint_path}")
    
    device = get_device(device)
    print_device_info(device)
    
    # 加载模型
    draft_model, target_model, tokenizer, knowledge_cache_manager = load_model(
        config_path, checkpoint_path, device=device
    )
    
    # 测试提示
    test_prompts = [
        "深度学习是",
        "自然语言处理是",
        "Transformer架构是",
        "机器学习是",
    ]
    
    print("\n" + "="*70)
    print("生成对比测试")
    print("="*70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(test_prompts)}: {prompt}")
        print(f"{'='*70}")
        
        # 检查知识检索
        if knowledge_cache_manager:
            result = knowledge_cache_manager.retrieve_by_similarity(prompt, top_k=1)
            if result:
                print(f"✓ 检索到相关知识")
        
        # 草稿模型生成
        print(f"\n[草稿模型]")
        start_time = time.time()
        draft_text = generate_text(draft_model, tokenizer, prompt, max_new_tokens=50)
        draft_time = time.time() - start_time
        print(f"生成时间: {draft_time:.2f}秒")
        print(f"生成文本: {draft_text}")
        
        # 目标模型生成
        print(f"\n[目标模型]")
        start_time = time.time()
        target_text = generate_text(target_model, tokenizer, prompt, max_new_tokens=50)
        target_time = time.time() - start_time
        print(f"生成时间: {target_time:.2f}秒")
        print(f"生成文本: {target_text}")
        
        if draft_time > 0:
            speedup = target_time / draft_time
            print(f"\n速度提升: {speedup:.2f}x")
    
    print("\n" + "="*70)
    print("推理测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

