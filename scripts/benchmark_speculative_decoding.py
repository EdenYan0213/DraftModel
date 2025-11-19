#!/usr/bin/env python3
"""
推测解码 vs 直接生成 时间对比测试
"""

import os
import sys
import torch
import yaml
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_enhanced_draft import Qwen3DraftModel
from models.knowledge_cache import KnowledgeCacheManager

def load_models(config_path: str, checkpoint_path: str = None):
    """加载模型"""
    print("="*70)
    print("加载模型")
    print("="*70)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载基础模型
    print("\n1. 加载基础模型...")
    loader = Qwen3Loader(config_path)
    target_model = loader.load_target_model(device='cpu')
    tokenizer = loader.load_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载知识缓存
    print("\n2. 加载知识缓存...")
    cache_path = "output/knowledge_cache/knowledge_cache.pth"
    knowledge_cache_manager = None
    
    if os.path.exists(cache_path):
        # 使用weights_only=False以支持numpy数组（embeddings）
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        
        # 获取配置
        knowledge_config = config.get('knowledge_enhancement', {})
        use_vector_retrieval = knowledge_config.get('use_vector_retrieval', True)
        embedding_model_name = knowledge_config.get('embedding_model_name', None)
        
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=config['base_model']['hidden_size'],
            num_heads=config['base_model']['num_attention_heads'],
            cache_dim=config['knowledge_enhancement']['cache_dim'],
            use_vector_retrieval=use_vector_retrieval,
            embedding_model_name=embedding_model_name
        )
        knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})
        
        # 恢复embeddings（如果存在）
        if 'knowledge_embeddings' in cache_data:
            knowledge_cache_manager.knowledge_embeddings = cache_data['knowledge_embeddings']
        
        print(f"✓ 知识缓存加载完成，共 {len(knowledge_cache_manager.kv_cache)} 个知识项")
        print(f"  检索方法: {knowledge_cache_manager.get_retrieval_method()}")
    
    # 创建草稿模型
    print("\n3. 创建草稿模型...")
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
    draft_model = draft_model.cpu()
    draft_model.eval()
    target_model.eval()
    
    # 加载训练好的权重
    if checkpoint_path is None:
        checkpoint_dir = "output/checkpoints"
        knowledge_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                                if 'knowledge' in f and 'best' in f and f.endswith('.pth')]
        if knowledge_checkpoints:
            knowledge_checkpoints.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
            checkpoint_path = os.path.join(checkpoint_dir, knowledge_checkpoints[0])
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n4. 加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        draft_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型权重加载完成")
    else:
        print(f"\n⚠ 未找到训练好的模型，使用未训练的模型")
    
    return draft_model, target_model, tokenizer

def direct_generation(target_model, tokenizer, prompt: str, max_new_tokens: int = 5, num_runs: int = 5):
    """
    方案1: 直接使用基础模型生成
    """
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_mask = (input_ids != pad_token_id).long()
    
    times = []
    generated_tokens = []
    
    for run in range(num_runs):
        current_input = input_ids.clone()
        current_mask = attention_mask.clone()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = target_model(current_input, attention_mask=current_mask)
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1).item()
                
                current_input = torch.cat([current_input, torch.tensor([[next_token_id]])], dim=1)
                current_mask = torch.ones((1, current_input.shape[1]), dtype=torch.long)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        times.append(elapsed)
        generated_tokens.append([tokenizer.decode([tid]) for tid in current_input[0][input_ids.shape[1]:].tolist()])
    
    avg_time = sum(times) / len(times)
    total_tokens = max_new_tokens * num_runs
    
    return {
        'avg_time': avg_time,
        'total_time': sum(times),
        'times': times,
        'tokens_per_second': total_tokens / sum(times),
        'generated_tokens': generated_tokens[0]  # 返回第一次生成的tokens
    }

def speculative_decoding(draft_model, target_model, tokenizer, prompt: str, max_new_tokens: int = 5, num_runs: int = 5, gamma: int = 5):
    """
    方案2: 并行推测解码 - 草稿模型一次性生成多个token，目标模型并行验证
    
    Args:
        gamma: 草稿模型一次性生成的token数量（并行度）
    """
    from inference.speculative_decoder import SpeculativeDecoder
    
    # 使用SpeculativeDecoder
    decoder = SpeculativeDecoder(draft_model, target_model, tokenizer, gamma=gamma)
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_mask = (input_ids != pad_token_id).long()
    
    times = []
    generated_tokens = []
    acceptance_counts = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            # 使用并行推测解码生成
            generated_sequence = decoder.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=0.9
            )
            
            # 计算接受率（简化：通过比较生成序列长度估算）
            # 实际应该从decoder内部获取，这里先简化
            generated_tokens_list = generated_sequence[0].tolist()
            generated_tokens_list = [tid for tid in generated_tokens_list if tid != pad_token_id]
            accepted_count = len(generated_tokens_list)  # 简化估算
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        times.append(elapsed)
        acceptance_counts.append(accepted_count)
        generated_tokens.append([tokenizer.decode([tid]) for tid in generated_tokens_list[:max_new_tokens]])
    
    avg_time = sum(times) / len(times)
    total_tokens = max_new_tokens * num_runs
    avg_acceptance = sum(acceptance_counts) / len(acceptance_counts)
    
    return {
        'avg_time': avg_time,
        'total_time': sum(times),
        'times': times,
        'tokens_per_second': total_tokens / sum(times),
        'generated_tokens': generated_tokens[0],
        'avg_acceptance_rate': avg_acceptance / max_new_tokens,
        'acceptance_counts': acceptance_counts
    }

def benchmark(prompts: List[str], max_new_tokens: int = 5, num_runs: int = 5):
    """基准测试"""
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 加载模型
    draft_model, target_model, tokenizer = load_models(config_path)
    
    print("\n" + "="*70)
    print("基准测试: 推测解码 vs 直接生成")
    print("="*70)
    print(f"\n测试配置:")
    print(f"  - 测试问题数量: {len(prompts)}")
    print(f"  - 每个问题生成token数: {max_new_tokens}")
    print(f"  - 每个问题运行次数: {num_runs}")
    print(f"  - 总测试次数: {len(prompts) * num_runs * 2}")
    
    results = {
        'direct': [],
        'speculative': []
    }
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(prompts)}: {prompt}")
        print(f"{'='*70}")
        
        # 方案1: 直接生成
        print("\n[方案1] 直接使用基础模型生成...")
        direct_result = direct_generation(target_model, tokenizer, prompt, max_new_tokens, num_runs)
        print(f"  平均时间: {direct_result['avg_time']:.4f}s")
        print(f"  生成速度: {direct_result['tokens_per_second']:.2f} tokens/s")
        print(f"  生成tokens: {''.join(direct_result['generated_tokens'])}")
        
        # 方案2: 并行推测解码
        print("\n[方案2] 并行推测解码（草稿模型并行生成 + 目标模型并行验证）...")
        speculative_result = speculative_decoding(draft_model, target_model, tokenizer, prompt, max_new_tokens, num_runs, gamma=5)
        print(f"  平均时间: {speculative_result['avg_time']:.4f}s")
        print(f"  生成速度: {speculative_result['tokens_per_second']:.2f} tokens/s")
        print(f"  平均接受率: {speculative_result['avg_acceptance_rate']:.2%}")
        print(f"  生成tokens: {''.join(speculative_result['generated_tokens'])}")
        
        # 计算加速比
        speedup = direct_result['avg_time'] / speculative_result['avg_time']
        print(f"\n  加速比: {speedup:.2f}x {'(更快)' if speedup > 1 else '(更慢)'}")
        
        results['direct'].append(direct_result)
        results['speculative'].append(speculative_result)
    
    # 总体汇总
    print("\n\n" + "="*70)
    print("总体汇总")
    print("="*70)
    
    direct_avg_time = sum(r['avg_time'] for r in results['direct']) / len(results['direct'])
    speculative_avg_time = sum(r['avg_time'] for r in results['speculative']) / len(results['speculative'])
    
    direct_avg_speed = sum(r['tokens_per_second'] for r in results['direct']) / len(results['direct'])
    speculative_avg_speed = sum(r['tokens_per_second'] for r in results['speculative']) / len(results['speculative'])
    
    overall_speedup = direct_avg_time / speculative_avg_time
    overall_acceptance = sum(r['avg_acceptance_rate'] for r in results['speculative']) / len(results['speculative'])
    
    print(f"\n[方案1] 直接生成:")
    print(f"  平均时间: {direct_avg_time:.4f}s")
    print(f"  平均速度: {direct_avg_speed:.2f} tokens/s")
    
    print(f"\n[方案2] 推测解码:")
    print(f"  平均时间: {speculative_avg_time:.4f}s")
    print(f"  平均速度: {speculative_avg_speed:.2f} tokens/s")
    print(f"  平均接受率: {overall_acceptance:.2%}")
    
    print(f"\n总体加速比: {overall_speedup:.2f}x {'(更快)' if overall_speedup > 1 else '(更慢)'}")
    
    # 详细对比表
    print(f"\n{'='*70}")
    print("详细对比表")
    print(f"{'='*70}")
    print(f"{'问题':<30} {'直接生成(s)':<15} {'推测解码(s)':<15} {'加速比':<10} {'接受率':<10}")
    print("-" * 70)
    for i, (prompt, direct_r, spec_r) in enumerate(zip(prompts, results['direct'], results['speculative']), 1):
        speedup = direct_r['avg_time'] / spec_r['avg_time']
        print(f"{prompt[:28]:<30} {direct_r['avg_time']:<15.4f} {spec_r['avg_time']:<15.4f} {speedup:<10.2f}x {spec_r['avg_acceptance_rate']:<10.2%}")
    
    return results

def main():
    """主函数"""
    # 测试问题
    test_prompts = [
        "深度学习是",
        "自然语言处理是",
        "Transformer架构是",
        "预训练语言模型是",
        "强化学习是"
    ]
    
    # 运行基准测试
    results = benchmark(test_prompts, max_new_tokens=5, num_runs=5)
    
    print("\n" + "="*70)
    print("基准测试完成！")
    print("="*70)

if __name__ == "__main__":
    main()

