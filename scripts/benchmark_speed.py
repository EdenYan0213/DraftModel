#!/usr/bin/env python3
"""
对比草稿模型和原模型的生成速度
"""

import os
import sys
import time
import torch
import yaml
from pathlib import Path
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_enhanced_draft import Qwen3DraftModel

def generate_with_model(model, tokenizer, input_ids, attention_mask, max_new_tokens=20, num_runs=5):
    """使用模型生成文本并测量时间"""
    model.eval()
    
    # 确保在CPU上（避免MPS问题）
    device = torch.device('cpu')
    model = model.to(device)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    times = []
    
    for i in range(num_runs):
        with torch.no_grad():
            start_time = time.time()
            
            try:
                # 使用模型的generate方法
                if hasattr(model, 'generate'):
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # 使用贪心解码以获得可重复的结果
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        attention_mask=attention_mask
                    )
                else:
                    # 如果没有generate方法，手动生成
                    current_input = input_ids
                    for _ in range(max_new_tokens):
                        outputs = model(
                            input_ids=current_input,
                            attention_mask=attention_mask
                        )
                        next_token_logits = outputs['logits'][:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        current_input = torch.cat([current_input, next_token], dim=1)
                    generated = current_input
                
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                
            except Exception as e:
                print(f"  ⚠ 运行 {i+1} 出错: {e}")
                continue
    
    if not times:
        return None, None, None
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    tokens_generated = generated.shape[1] - input_ids.shape[1]
    
    return avg_time, std_time, tokens_generated

def main():
    print("="*70)
    print("草稿模型 vs 原模型 - 生成速度对比测试")
    print("="*70)
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    with open('configs/qwen3_0.6b_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("✓ 配置加载完成")
    
    # 2. 加载基础模型和tokenizer
    print("\n2. 加载基础模型和tokenizer...")
    loader = Qwen3Loader('configs/qwen3_0.6b_config.yaml')
    target_model = loader.load_target_model(device='cpu')
    tokenizer = loader.load_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ 基础模型加载完成")
    
    # 3. 加载知识缓存（如果存在）
    print("\n3. 加载知识缓存...")
    cache_path = "output/knowledge_cache/knowledge_cache.pth"
    knowledge_cache_manager = None
    
    if os.path.exists(cache_path):
        from models.knowledge_cache import KnowledgeCacheManager
        cache_data = torch.load(cache_path, map_location='cpu')
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=config['base_model']['hidden_size'],
            num_heads=config['base_model']['num_attention_heads'],
            cache_dim=config['knowledge_enhancement']['cache_dim']
        )
        knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})
        knowledge_cache_manager.compressed_cache = cache_data.get('compressed_cache', {})
        print(f"✓ 知识缓存加载完成，共 {len(knowledge_cache_manager.kv_cache)} 个知识项")
    else:
        print("⚠ 知识缓存不存在，将不使用知识增强")
    
    # 4. 创建并加载草稿模型
    print("\n4. 创建并加载草稿模型...")
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
    draft_model = draft_model.cpu()
    draft_model.eval()
    
    # 加载训练好的权重
    checkpoint_path = "output/checkpoints/best_draft_model_epoch5.pth"
    if not os.path.exists(checkpoint_path):
        # 查找其他best模型
        checkpoint_files = [f for f in os.listdir('output/checkpoints/') 
                           if 'best' in f and f.endswith('.pth')]
        if checkpoint_files:
            checkpoint_path = os.path.join('output/checkpoints/', sorted(checkpoint_files)[-1])
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        draft_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 草稿模型权重加载完成 ({checkpoint_path})")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', 0):.4f}")
    else:
        print("⚠ 未找到训练好的模型，使用未训练的模型")
    
    # 5. 准备测试数据
    print("\n5. 准备测试数据...")
    test_prompts = [
        "今天天气很好，",
        "人工智能的未来发展",
        "深度学习模型在自然语言处理中",
        "中国的首都是",
        "自然语言处理是"
    ]
    
    max_new_tokens = 20
    num_runs = 3  # 每个模型运行3次取平均
    
    print(f"测试配置:")
    print(f"  - 测试提示数: {len(test_prompts)}")
    print(f"  - 每个提示生成token数: {max_new_tokens}")
    print(f"  - 每个测试运行次数: {num_runs}")
    
    # 6. 速度对比测试
    print("\n" + "="*70)
    print("开始速度对比测试")
    print("="*70)
    
    results = {
        'draft': {'times': [], 'tokens': []},
        'target': {'times': [], 'tokens': []}
    }
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- 测试 {i}/{len(test_prompts)}: {prompt[:30]}... ---")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids']
        
        # 创建attention_mask
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        attention_mask = (input_ids != pad_token_id).long()
        
        # 测试草稿模型
        print("  测试草稿模型...")
        draft_time, draft_std, draft_tokens = generate_with_model(
            draft_model, tokenizer, input_ids, attention_mask, 
            max_new_tokens=max_new_tokens, num_runs=num_runs
        )
        
        if draft_time is not None:
            results['draft']['times'].append(draft_time)
            results['draft']['tokens'].append(draft_tokens)
            print(f"    平均时间: {draft_time:.4f}s ± {draft_std:.4f}s")
            print(f"    生成token数: {draft_tokens}")
            print(f"    速度: {draft_tokens/draft_time:.2f} tokens/s")
        
        # 测试目标模型
        print("  测试目标模型...")
        target_time, target_std, target_tokens = generate_with_model(
            target_model, tokenizer, input_ids, attention_mask,
            max_new_tokens=max_new_tokens, num_runs=num_runs
        )
        
        if target_time is not None:
            results['target']['times'].append(target_time)
            results['target']['tokens'].append(target_tokens)
            print(f"    平均时间: {target_time:.4f}s ± {target_std:.4f}s")
            print(f"    生成token数: {target_tokens}")
            print(f"    速度: {target_tokens/target_time:.2f} tokens/s")
        
        # 计算加速比
        if draft_time is not None and target_time is not None:
            speedup = target_time / draft_time
            print(f"    加速比: {speedup:.2f}x")
    
    # 7. 汇总结果
    print("\n" + "="*70)
    print("速度对比汇总")
    print("="*70)
    
    if results['draft']['times'] and results['target']['times']:
        draft_avg_time = statistics.mean(results['draft']['times'])
        target_avg_time = statistics.mean(results['target']['times'])
        
        draft_avg_tokens = statistics.mean(results['draft']['tokens'])
        target_avg_tokens = statistics.mean(results['target']['tokens'])
        
        draft_speed = draft_avg_tokens / draft_avg_time
        target_speed = target_avg_tokens / target_avg_time
        
        speedup = target_avg_time / draft_avg_time
        
        print(f"\n草稿模型:")
        print(f"  - 平均生成时间: {draft_avg_time:.4f}s")
        print(f"  - 平均生成token数: {draft_avg_tokens:.1f}")
        print(f"  - 平均生成速度: {draft_speed:.2f} tokens/s")
        
        print(f"\n目标模型:")
        print(f"  - 平均生成时间: {target_avg_time:.4f}s")
        print(f"  - 平均生成token数: {target_avg_tokens:.1f}")
        print(f"  - 平均生成速度: {target_speed:.2f} tokens/s")
        
        print(f"\n性能对比:")
        print(f"  - 时间加速比: {speedup:.2f}x")
        print(f"  - 速度提升: {(speedup-1)*100:.1f}%")
        
        if speedup > 1:
            print(f"\n✓ 草稿模型比目标模型快 {speedup:.2f} 倍！")
        else:
            print(f"\n⚠ 草稿模型比目标模型慢 {1/speedup:.2f} 倍")
            print("  可能原因：模型未充分训练或结构问题")
    else:
        print("\n⚠ 无法计算汇总结果（部分测试失败）")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

if __name__ == "__main__":
    main()

