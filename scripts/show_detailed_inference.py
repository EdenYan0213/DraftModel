#!/usr/bin/env python3
"""
详细展示草稿模型每次生成的token和基础模型的接受情况
"""

import os
import sys
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from typing import List, Dict, Tuple

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
    
    return target_model, draft_model, tokenizer, knowledge_cache_manager

def show_detailed_inference(target_model, draft_model, tokenizer, knowledge_cache_manager, 
                           prompt: str, num_draft_tokens: int = 5, acceptance_threshold: float = 0.05):
    """详细展示推理过程"""
    print("\n" + "="*70)
    print(f"详细推理展示: {prompt}")
    print("="*70)
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs['input_ids'].clone()
    current_input = input_ids.clone()
    
    print(f"\n输入提示: {prompt}")
    print(f"输入token数: {input_ids.shape[1]}")
    print(f"\n草稿模型将生成 {num_draft_tokens} 个token，然后基础模型验证...")
    print("-"*70)
    
    # 草稿模型生成token序列
    draft_tokens = []
    draft_token_texts = []
    draft_probs = []
    
    print("\n【步骤1】草稿模型生成token序列:")
    print("-"*70)
    
    # 一次性检索知识（避免重复检索）
    retrieved_knowledge = None
    if knowledge_cache_manager is not None:
        retrieved_knowledge = knowledge_cache_manager.retrieve_by_similarity(
            query=prompt,
            topk=1
        )
    
    for step in range(num_draft_tokens):
        with torch.no_grad():
            # 草稿模型前向传播
            outputs = draft_model.forward(
                current_input,
                retrieve_knowledge=(step == 0),  # 只在第一步检索
                query_text=prompt if step == 0 else None
            )
            
            logits = outputs['logits'][:, -1, :]  # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            
            # 获取top-3预测
            top_probs, top_indices = torch.topk(probs, k=3, dim=-1)
            
            # 贪心选择
            next_token_id = torch.argmax(logits, dim=-1).item()
            next_token_prob = probs[0, next_token_id].item()
            
            # 解码token
            next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
            
            draft_tokens.append(next_token_id)
            draft_token_texts.append(next_token_text)
            draft_probs.append(next_token_prob)
            
            # 显示草稿模型预测
            print(f"\n  位置 {step + 1}:")
            print(f"    草稿模型预测: '{next_token_text}' (ID: {next_token_id}, 概率: {next_token_prob:.4f})")
            print(f"    Top-3预测:")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                token_text = tokenizer.decode([idx.item()], skip_special_tokens=False)
                print(f"      {i+1}. '{token_text}' (ID: {idx.item()}, 概率: {prob.item():.4f})")
            
            # 更新输入
            current_input = torch.cat([current_input, torch.tensor([[next_token_id]])], dim=1)
    
    print("\n" + "-"*70)
    print(f"\n草稿模型生成的完整序列: {''.join(draft_token_texts)}")
    print("-"*70)
    
    # 基础模型验证
    print("\n【步骤2】基础模型验证草稿模型生成的token:")
    print("-"*70)
    
    accepted_tokens = []
    accepted_token_texts = []
    rejected_at = None
    
    # 重置输入
    current_target_input = input_ids.clone()
    
    for step, (draft_token_id, draft_token_text) in enumerate(zip(draft_tokens, draft_token_texts)):
        with torch.no_grad():
            # 基础模型预测
            target_outputs = target_model(current_target_input)
            target_logits = target_outputs.logits[:, -1, :]
            target_probs = F.softmax(target_logits, dim=-1)
            
            # 获取基础模型的top-3预测
            target_top_probs, target_top_indices = torch.topk(target_probs, k=3, dim=-1)
            
            # 基础模型的贪心预测
            target_token_id = torch.argmax(target_logits, dim=-1).item()
            target_token_text = tokenizer.decode([target_token_id], skip_special_tokens=False)
            
            # 草稿模型token在基础模型中的概率
            draft_token_prob_in_target = target_probs[0, draft_token_id].item()
            
            # 判断是否接受
            # 条件1: 草稿模型预测 == 基础模型预测
            # 条件2: 草稿模型token在基础模型中的概率 > 阈值
            # 条件3: 草稿模型token在基础模型的Top-3中
            is_exact_match = (draft_token_id == target_token_id)
            is_prob_high = (draft_token_prob_in_target > acceptance_threshold)
            is_in_top3 = (draft_token_id in target_top_indices[0].tolist())
            
            accept = is_exact_match or is_prob_high or is_in_top3
            
            print(f"\n  验证位置 {step + 1} (草稿模型预测: '{draft_token_text}')")
            print(f"    基础模型预测: '{target_token_text}' (ID: {target_token_id}, 概率: {target_probs[0, target_token_id].item():.4f})")
            print(f"    草稿token在基础模型中的概率: {draft_token_prob_in_target:.4f}")
            print(f"    基础模型Top-3预测:")
            for i, (prob, idx) in enumerate(zip(target_top_probs[0], target_top_indices[0])):
                token_text = tokenizer.decode([idx.item()], skip_special_tokens=False)
                marker = " ← 草稿模型预测" if idx.item() == draft_token_id else ""
                print(f"      {i+1}. '{token_text}' (ID: {idx.item()}, 概率: {prob.item():.4f}){marker}")
            
            print(f"\n    接受判断:")
            print(f"      - 完全匹配: {is_exact_match} (草稿 == 基础)")
            print(f"      - 概率足够: {is_prob_high} (概率 > {acceptance_threshold})")
            print(f"      - Top-3包含: {is_in_top3}")
            print(f"      → 最终决定: {'✓ 接受' if accept else '✗ 拒绝'}")
            
            if accept:
                accepted_tokens.append(draft_token_id)
                accepted_token_texts.append(draft_token_text)
                current_target_input = torch.cat([current_target_input, torch.tensor([[draft_token_id]])], dim=1)
            else:
                rejected_at = step + 1
                print(f"\n    ⚠️  在位置 {rejected_at} 被拒绝，基础模型将从此位置开始自己生成")
                break
    
    print("\n" + "-"*70)
    print("\n【步骤3】接受结果汇总:")
    print("-"*70)
    print(f"  草稿模型生成: {num_draft_tokens} 个token")
    print(f"  基础模型接受: {len(accepted_tokens)} 个token")
    print(f"  接受率: {len(accepted_tokens) / num_draft_tokens * 100:.2f}%")
    
    if accepted_tokens:
        print(f"\n  接受的token序列: {''.join(accepted_token_texts)}")
    
    if rejected_at:
        print(f"\n  在位置 {rejected_at} 被拒绝")
        print(f"  基础模型将从位置 {rejected_at} 开始自己生成剩余token")
    else:
        print(f"\n  ✓ 所有 {num_draft_tokens} 个token都被接受！")
    
    # 如果被拒绝，展示基础模型如何继续生成
    if rejected_at:
        print("\n" + "-"*70)
        print(f"\n【步骤4】基础模型从位置 {rejected_at} 开始生成:")
        print("-"*70)
        
        remaining_tokens = []
        remaining_token_texts = []
        max_remaining = 5  # 只展示前5个
        
        for step in range(max_remaining):
            with torch.no_grad():
                target_outputs = target_model(current_target_input)
                target_logits = target_outputs.logits[:, -1, :]
                target_probs = F.softmax(target_logits, dim=-1)
                
                next_token_id = torch.argmax(target_logits, dim=-1).item()
                next_token_prob = target_probs[0, next_token_id].item()
                next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
                
                remaining_tokens.append(next_token_id)
                remaining_token_texts.append(next_token_text)
                
                print(f"  位置 {rejected_at + step}: '{next_token_text}' (ID: {next_token_id}, 概率: {next_token_prob:.4f})")
                
                current_target_input = torch.cat([current_target_input, torch.tensor([[next_token_id]])], dim=1)
        
        print(f"\n  基础模型生成的序列: {''.join(remaining_token_texts)}")
    
    print("\n" + "="*70)
    
    return {
        'draft_tokens': draft_tokens,
        'draft_token_texts': draft_token_texts,
        'accepted_tokens': accepted_tokens,
        'accepted_token_texts': accepted_token_texts,
        'rejected_at': rejected_at,
        'acceptance_rate': len(accepted_tokens) / num_draft_tokens if num_draft_tokens > 0 else 0
    }

def main():
    """主函数"""
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 加载模型
    target_model, draft_model, tokenizer, knowledge_cache_manager = load_models(config_path)
    
    # 测试问题
    test_prompts = [
        "深度学习是",
        "自然语言处理是"
    ]
    
    print("\n" + "="*70)
    print("开始详细推理展示")
    print("="*70)
    
    results = []
    for i, prompt in enumerate(test_prompts, 1):
        result = show_detailed_inference(
            target_model, draft_model, tokenizer, knowledge_cache_manager,
            prompt=prompt,
            num_draft_tokens=5,
            acceptance_threshold=0.05
        )
        results.append(result)
        
        if i < len(test_prompts):
            print("\n\n")
    
    # 汇总
    print("\n" + "="*70)
    print("总体汇总")
    print("="*70)
    avg_acceptance = sum(r['acceptance_rate'] for r in results) / len(results) if results else 0
    print(f"\n平均接受率: {avg_acceptance * 100:.2f}%")
    
    for i, (prompt, result) in enumerate(zip(test_prompts, results), 1):
        print(f"\n测试 {i}: {prompt}")
        print(f"  接受率: {result['acceptance_rate'] * 100:.2f}%")
        print(f"  接受token数: {len(result['accepted_tokens'])}/5")
        if result['rejected_at']:
            print(f"  拒绝位置: {result['rejected_at']}")

if __name__ == "__main__":
    main()

