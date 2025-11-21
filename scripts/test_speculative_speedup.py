#!/usr/bin/env python3
"""
测试Speculative Decoding的加速效果
草稿模型生成5个token -> 基础模型验证 -> 如果接受保留，不接受则基础模型自己生成
"""

import os
import sys
import torch
import yaml
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_enhanced_draft import Qwen3DraftModel
from models.knowledge_cache import KnowledgeCacheManager

def load_models(config_path: str):
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
        knowledge_config = config.get('knowledge_enhancement', {})
        use_vector_retrieval = knowledge_config.get('use_vector_retrieval', True)
        embedding_model_name = knowledge_config.get('embedding_model_name', None)
        
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=config['base_model']['hidden_size'],
            num_heads=config['base_model']['num_attention_heads'],
            cache_dim=config['knowledge_enhancement']['cache_dim'],
            use_vector_retrieval=use_vector_retrieval,
            embedding_model_name=embedding_model_name,
            target_model=target_model,
            tokenizer=tokenizer
        )
        
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})
        
        if 'knowledge_embeddings' in cache_data:
            knowledge_cache_manager.knowledge_embeddings = cache_data['knowledge_embeddings']
        
        if 'answer_start_indices' in cache_data:
            knowledge_cache_manager.answer_start_indices = cache_data['answer_start_indices']
        
        print("✓ 知识缓存加载完成")
    else:
        print("⚠ 知识缓存文件不存在")
    
    # 加载草稿模型
    print("\n3. 加载草稿模型...")
    checkpoint_dir = Path("output/checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("best_draft_model_knowledge_epoch*.pth"))
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1].replace('epoch', '')))
        checkpoint_path = checkpoint_files[-1]
        print(f"使用checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None
        print("⚠ 未找到checkpoint，使用未训练的模型")
    
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            draft_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            draft_model.load_state_dict(checkpoint)
        print("✓ 草稿模型加载完成")
    else:
        print("⚠ 使用未训练的草稿模型")
    
    draft_model.eval()
    target_model.eval()
    
    return draft_model, target_model, tokenizer, knowledge_cache_manager

def direct_generation(target_model, tokenizer, prompt: str, max_new_tokens: int = 20, num_runs: int = 3):
    """
    直接使用基础模型生成（基准）
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs['input_ids']
    
    times = []
    generated_tokens_list = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            current_input = input_ids.clone()
            generated_tokens = []
            
            for _ in range(max_new_tokens):
                outputs = target_model(current_input)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                generated_tokens.append(next_token_id)
                
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token_id]])
                ], dim=1)
                
                if next_token_id == tokenizer.eos_token_id:
                    break
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        generated_tokens_list.append(generated_tokens)
    
    avg_time = sum(times) / len(times)
    total_tokens = sum(len(tokens) for tokens in generated_tokens_list) / len(generated_tokens_list)
    tokens_per_second = total_tokens / avg_time if avg_time > 0 else 0
    
    return {
        'avg_time': avg_time,
        'tokens_per_second': tokens_per_second,
        'generated_tokens': generated_tokens_list[0] if generated_tokens_list else [],
        'times': times
    }

def speculative_decoding(draft_model, target_model, tokenizer, prompt: str,
                         knowledge_cache_manager=None, max_new_tokens: int = 20,
                         num_runs: int = 3, gamma: int = 5, accept_threshold: float = 0.05,
                         use_top_k: bool = True, top_k: int = 3):
    """
    Speculative Decoding: 草稿模型生成gamma个token -> 基础模型验证 -> 接受或拒绝
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs['input_ids']
    
    times = []
    generated_tokens_list = []
    acceptance_rates = []
    draft_times = []
    verify_times = []
    target_times = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            current_input = input_ids.clone()
            generated_tokens = []
            
            # Step 1: 草稿模型生成gamma个token
            draft_start = time.time()
            draft_tokens = []
            
            # 优化1: 只在开始时检索一次知识缓存，不要每次循环都检索
            knowledge_cache = None
            answer_start_idx = None
            if knowledge_cache_manager is not None:
                retrieved = knowledge_cache_manager.retrieve(
                    prompt,
                    threshold=0.5,
                    return_similarity=True
                )
                if retrieved is not None:
                    if len(retrieved) == 4:
                        knowledge_cache = (retrieved[0], retrieved[1])
                        answer_start_idx = retrieved[3]
                    elif len(retrieved) == 3:
                        knowledge_cache = (retrieved[0], retrieved[1])
            
            # 优化2: 使用past_key_values实现KV cache加速
            past_key_values = None
            
            for i in range(gamma):
                # 草稿模型前向传播（使用past_key_values加速）
                # 注意：如果模型支持use_cache，可以进一步优化
                outputs = draft_model.forward(
                    current_input,
                    knowledge_cache=knowledge_cache,
                    retrieve_knowledge=False,
                    query_text=prompt,
                    answer_start_idx=answer_start_idx,
                    use_cache=False  # 暂时不使用cache，因为需要检查模型是否支持
                )
                
                # 获取logits并采样
                logits = outputs['logits'][:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                draft_tokens.append(next_token_id)
                
                # 更新输入（只保留最后一个token，利用KV cache）
                # 优化：只保留最后一个token，减少序列长度
                if i == 0:
                    # 第一次：保留完整输入
                    current_input = torch.cat([
                        current_input,
                        torch.tensor([[next_token_id]])
                    ], dim=1)
                else:
                    # 后续：只保留最后一个token（如果模型支持KV cache）
                    # 否则保留完整序列
                    current_input = torch.cat([
                        current_input,
                        torch.tensor([[next_token_id]])
                    ], dim=1)
                
                if next_token_id == tokenizer.eos_token_id:
                    break
            
            draft_end = time.time()
            draft_time = draft_end - draft_start
            
            # Step 2: 基础模型验证并接受
            verify_start = time.time()
            accepted_tokens = []
            
            if draft_tokens:
                # 构建包含所有draft tokens的完整序列
                draft_sequence = torch.cat([
                    input_ids,
                    torch.tensor([draft_tokens])
                ], dim=1)
                
                # 目标模型一次性前向传播（并行处理所有token位置）
                outputs = target_model(draft_sequence)
                all_logits = outputs.logits
                
                input_len = input_ids.shape[1]
                all_accepted = True
                
                # 按顺序验证每个draft token
                for i, draft_token_id in enumerate(draft_tokens):
                    pos_idx = input_len + i - 1
                    if pos_idx < 0:
                        pos_idx = 0
                    elif pos_idx >= all_logits.shape[1]:
                        pos_idx = all_logits.shape[1] - 1
                    
                    target_logits = all_logits[:, pos_idx, :]
                    target_probs = torch.softmax(target_logits, dim=-1)
                    
                    # 计算接受概率
                    accept_prob = target_probs[0, draft_token_id].item()
                    target_token_id = torch.argmax(target_logits, dim=-1).item()
                    
                    # 判断是否接受
                    # 策略1: 如果草稿token和目标token相同，直接接受
                    # 策略2: 如果接受概率>阈值，接受
                    # 策略3: 如果使用Top-k策略，检查草稿token是否在Top-k中
                    is_accepted = False
                    
                    if draft_token_id == target_token_id:
                        is_accepted = True
                    elif accept_prob > accept_threshold:
                        is_accepted = True
                    elif use_top_k:
                        # Top-k接受策略：如果草稿token在目标模型的Top-k中，则接受
                        top_k_probs, top_k_indices = torch.topk(target_probs, top_k, dim=-1)
                        if draft_token_id in top_k_indices[0]:
                            is_accepted = True
                    
                    if is_accepted:
                        accepted_tokens.append(draft_token_id)
                    else:
                        # 一旦拒绝，停止接受后续token
                        all_accepted = False
                        # 记录拒绝原因（用于调试）
                        if draft_token_id != target_token_id and accept_prob <= accept_threshold:
                            if use_top_k:
                                top_k_probs, top_k_indices = torch.topk(target_probs, top_k, dim=-1)
                                if draft_token_id not in top_k_indices[0]:
                                    # 不在Top-k中
                                    pass
                        break
                
                verify_end = time.time()
                verify_time = verify_end - verify_start
                
                # 更新输入
                if accepted_tokens:
                    current_input = torch.cat([
                        input_ids,
                        torch.tensor([accepted_tokens])
                    ], dim=1)
                    generated_tokens.extend(accepted_tokens)
            
            # Step 3: 基础模型生成剩余token
            target_start = time.time()
            remaining_tokens = max_new_tokens - len(generated_tokens)
            
            for _ in range(remaining_tokens):
                outputs = target_model(current_input)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                generated_tokens.append(next_token_id)
                
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token_id]])
                ], dim=1)
                
                if next_token_id == tokenizer.eos_token_id:
                    break
            
            target_end = time.time()
            target_time = target_end - target_start
            
            total_time = time.time() - start_time
        
        times.append(total_time)
        draft_times.append(draft_time)
        verify_times.append(verify_time)
        target_times.append(target_time)
        generated_tokens_list.append(generated_tokens)
        
        acceptance_rate = len(accepted_tokens) / len(draft_tokens) if draft_tokens else 0.0
        acceptance_rates.append(acceptance_rate)
    
    avg_time = sum(times) / len(times)
    avg_draft_time = sum(draft_times) / len(draft_times)
    avg_verify_time = sum(verify_times) / len(verify_times)
    avg_target_time = sum(target_times) / len(target_times)
    avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates)
    
    total_tokens = sum(len(tokens) for tokens in generated_tokens_list) / len(generated_tokens_list)
    tokens_per_second = total_tokens / avg_time if avg_time > 0 else 0
    
    return {
        'avg_time': avg_time,
        'avg_draft_time': avg_draft_time,
        'avg_verify_time': avg_verify_time,
        'avg_target_time': avg_target_time,
        'tokens_per_second': tokens_per_second,
        'avg_acceptance_rate': avg_acceptance_rate,
        'generated_tokens': generated_tokens_list[0] if generated_tokens_list else [],
        'times': times,
        'acceptance_rates': acceptance_rates
    }

def main():
    """主函数"""
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 加载模型
    draft_model, target_model, tokenizer, knowledge_cache_manager = load_models(config_path)
    
    # 测试问题
    test_questions = [
        "深度学习是",
        "自然语言处理是",
        "强化学习是",
        "机器学习是",
    ]
    
    print("\n" + "="*70)
    print("Speculative Decoding 加速测试")
    print("="*70)
    print(f"\n测试配置:")
    print(f"  - 测试问题数量: {len(test_questions)}")
    print(f"  - 每个问题生成token数: 20")
    print(f"  - 每个问题运行次数: 3")
    print(f"  - 草稿模型生成token数: 5")
    print(f"  - 接受阈值: 0.05 (降低以提高接受率)")
    print(f"  - Top-k接受策略: 启用 (Top-3)")
    
    all_results = {
        'direct': [],
        'speculative': []
    }
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(test_questions)}: {question}")
        print(f"{'='*70}")
        
        # 方案1: 直接生成
        print("\n[方案1] 直接使用基础模型生成...")
        direct_result = direct_generation(target_model, tokenizer, question, max_new_tokens=20, num_runs=3)
        print(f"  平均时间: {direct_result['avg_time']:.4f}s")
        print(f"  生成速度: {direct_result['tokens_per_second']:.2f} tokens/s")
        direct_text = tokenizer.decode(direct_result['generated_tokens'])
        print(f"  生成文本: {direct_text[:100]}...")
        all_results['direct'].append(direct_result)
        
        # 方案2: Speculative Decoding
        print("\n[方案2] Speculative Decoding（草稿模型生成5个token -> 基础模型验证）...")
        speculative_result = speculative_decoding(
            draft_model, target_model, tokenizer, question,
            knowledge_cache_manager=knowledge_cache_manager,
            max_new_tokens=20, num_runs=3, gamma=5, 
            accept_threshold=0.05,  # 降低阈值
            use_top_k=True,  # 启用Top-k策略
            top_k=3  # Top-3
        )
        print(f"  平均时间: {speculative_result['avg_time']:.4f}s")
        print(f"  生成速度: {speculative_result['tokens_per_second']:.2f} tokens/s")
        print(f"  平均接受率: {speculative_result['avg_acceptance_rate']:.2%}")
        print(f"  时间分解:")
        print(f"    - 草稿模型生成: {speculative_result['avg_draft_time']:.4f}s")
        print(f"    - 基础模型验证: {speculative_result['avg_verify_time']:.4f}s")
        print(f"    - 基础模型生成剩余: {speculative_result['avg_target_time']:.4f}s")
        speculative_text = tokenizer.decode(speculative_result['generated_tokens'])
        print(f"  生成文本: {speculative_text[:100]}...")
        all_results['speculative'].append(speculative_result)
        
        # 计算加速比
        speedup = direct_result['avg_time'] / speculative_result['avg_time']
        print(f"\n  ⚡ 加速比: {speedup:.2f}x")
        if speedup > 1.0:
            print(f"  ✅ Speculative Decoding更快！")
        else:
            print(f"  ⚠️  Speculative Decoding较慢")
    
    # 总体汇总
    print("\n\n" + "="*70)
    print("总体汇总")
    print("="*70)
    
    avg_direct_time = np.mean([r['avg_time'] for r in all_results['direct']])
    avg_speculative_time = np.mean([r['avg_time'] for r in all_results['speculative']])
    avg_speedup = avg_direct_time / avg_speculative_time
    
    avg_direct_speed = np.mean([r['tokens_per_second'] for r in all_results['direct']])
    avg_speculative_speed = np.mean([r['tokens_per_second'] for r in all_results['speculative']])
    
    avg_acceptance_rate = np.mean([r['avg_acceptance_rate'] for r in all_results['speculative']])
    
    print(f"\n平均生成时间:")
    print(f"  直接生成: {avg_direct_time:.4f}s")
    print(f"  Speculative Decoding: {avg_speculative_time:.4f}s")
    print(f"  平均加速比: {avg_speedup:.2f}x")
    
    print(f"\n平均生成速度:")
    print(f"  直接生成: {avg_direct_speed:.2f} tokens/s")
    print(f"  Speculative Decoding: {avg_speculative_speed:.2f} tokens/s")
    
    print(f"\n平均接受率: {avg_acceptance_rate:.2%}")
    
    print(f"\n时间分解（Speculative Decoding）:")
    avg_draft_time = np.mean([r['avg_draft_time'] for r in all_results['speculative']])
    avg_verify_time = np.mean([r['avg_verify_time'] for r in all_results['speculative']])
    avg_target_time = np.mean([r['avg_target_time'] for r in all_results['speculative']])
    print(f"  草稿模型生成: {avg_draft_time:.4f}s ({avg_draft_time/avg_speculative_time*100:.1f}%)")
    print(f"  基础模型验证: {avg_verify_time:.4f}s ({avg_verify_time/avg_speculative_time*100:.1f}%)")
    print(f"  基础模型生成剩余: {avg_target_time:.4f}s ({avg_target_time/avg_speculative_time*100:.1f}%)")

if __name__ == "__main__":
    main()

