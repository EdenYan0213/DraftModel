#!/usr/bin/env python3
"""
测试知识库中的问题 - 查看知识增强效果
"""

import os
import sys
import torch
import yaml
from pathlib import Path
from typing import List, Dict
import numpy as np

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
        cache_data = torch.load(cache_path, map_location='cpu')
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=config['base_model']['hidden_size'],
            num_heads=config['base_model']['num_attention_heads'],
            cache_dim=config['knowledge_enhancement']['cache_dim']
        )
        knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})
        print(f"✓ 知识缓存加载完成，共 {len(knowledge_cache_manager.kv_cache)} 个知识项")
    
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
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', 0):.4f}")
    else:
        print(f"\n⚠ 未找到训练好的模型，使用未训练的模型")
    
    return draft_model, target_model, tokenizer

def analyze_question(draft_model, target_model, tokenizer, question: str, max_new_tokens: int = 5):
    """分析单个问题"""
    print(f"\n{'='*70}")
    print(f"问题: {question}")
    print(f"{'='*70}")
    
    # Tokenize
    inputs = tokenizer(question, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_mask = (input_ids != pad_token_id).long()
    
    results = {
        'question': question,
        'tokens': [],
        'acceptance_probs': [],
        'is_accepted': []
    }
    
    current_input = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            print(f"\n--- Step {step + 1}/{max_new_tokens} ---")
            
            # 草稿模型推理
            draft_outputs = draft_model(
                current_input,
                attention_mask=attention_mask,
                retrieve_knowledge=True,
                query_text=question
            )
            draft_logits = draft_outputs['logits'][:, -1, :]
            draft_probs = torch.softmax(draft_logits, dim=-1)
            draft_token_id = torch.argmax(draft_logits, dim=-1).item()
            draft_token_text = tokenizer.decode([draft_token_id])
            draft_token_prob = draft_probs[0, draft_token_id].item()
            
            # 目标模型推理
            target_outputs = target_model(current_input, attention_mask=attention_mask)
            target_logits = target_outputs.logits[:, -1, :]
            target_probs = torch.softmax(target_logits, dim=-1)
            target_token_id = torch.argmax(target_logits, dim=-1).item()
            target_token_text = tokenizer.decode([target_token_id])
            target_token_prob = target_probs[0, target_token_id].item()
            
            # 接受概率
            acceptance_prob = target_probs[0, draft_token_id].item()
            is_accepted = (draft_token_id == target_token_id)
            
            print(f"草稿模型: {draft_token_text!r} (概率: {draft_token_prob:.4f})")
            print(f"目标模型: {target_token_text!r} (概率: {target_token_prob:.4f})")
            print(f"接受概率: {acceptance_prob:.6f} | 是否接受: {'✓' if is_accepted else '✗'}")
            
            # Top-3预测
            draft_top3 = torch.topk(draft_probs, 3, dim=-1)
            target_top3 = torch.topk(target_probs, 3, dim=-1)
            
            print(f"草稿Top-3: ", end="")
            for i in range(3):
                token_id = draft_top3.indices[0, i].item()
                prob = draft_top3.values[0, i].item()
                token_text = tokenizer.decode([token_id])
                print(f"{token_text!r}({prob:.3f}) ", end="")
            print()
            
            print(f"目标Top-3: ", end="")
            for i in range(3):
                token_id = target_top3.indices[0, i].item()
                prob = target_top3.values[0, i].item()
                token_text = tokenizer.decode([token_id])
                print(f"{token_text!r}({prob:.3f}) ", end="")
            print()
            
            results['tokens'].append({
                'step': step + 1,
                'draft_token': draft_token_text,
                'target_token': target_token_text,
                'acceptance_prob': acceptance_prob,
                'is_accepted': is_accepted
            })
            results['acceptance_probs'].append(acceptance_prob)
            results['is_accepted'].append(is_accepted)
            
            # 更新输入
            current_input = torch.cat([current_input, torch.tensor([[draft_token_id]])], dim=1)
            attention_mask = torch.ones((1, current_input.shape[1]), dtype=torch.long)
    
    # 计算统计
    acceptance_rate = sum(results['is_accepted']) / len(results['is_accepted']) if results['is_accepted'] else 0.0
    avg_acceptance_prob = np.mean(results['acceptance_probs'])
    
    results['acceptance_rate'] = acceptance_rate
    results['avg_acceptance_prob'] = avg_acceptance_prob
    
    print(f"\n{'='*70}")
    print(f"汇总")
    print(f"{'='*70}")
    print(f"接受率: {acceptance_rate:.2%} ({sum(results['is_accepted'])}/{len(results['is_accepted'])})")
    print(f"平均接受概率: {avg_acceptance_prob:.6f}")
    
    return results

def main():
    """主函数"""
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 加载模型
    draft_model, target_model, tokenizer = load_models(config_path)
    
    # 测试问题（这些都在知识库中）
    questions = [
        "深度学习是",
        "自然语言处理是",
        "Transformer架构是",
        "预训练语言模型是",
        "强化学习是"
    ]
    
    print("\n" + "="*70)
    print("知识库问题推理分析")
    print("="*70)
    print(f"\n测试问题数量: {len(questions)}")
    print("注意: 这些问题都在知识库中，应该能看到知识增强的效果")
    
    all_results = []
    
    for question in questions:
        results = analyze_question(draft_model, target_model, tokenizer, question, max_new_tokens=5)
        all_results.append(results)
    
    # 总体汇总
    print("\n\n" + "="*70)
    print("总体汇总")
    print("="*70)
    
    total_accepted = sum(sum(r['is_accepted']) for r in all_results)
    total_tokens = sum(len(r['is_accepted']) for r in all_results)
    overall_acceptance_rate = total_accepted / total_tokens if total_tokens > 0 else 0.0
    
    all_acceptance_probs = []
    for r in all_results:
        all_acceptance_probs.extend(r['acceptance_probs'])
    
    print(f"\n总体接受率: {overall_acceptance_rate:.2%} ({total_accepted}/{total_tokens})")
    print(f"平均接受概率: {np.mean(all_acceptance_probs):.6f}")
    print(f"接受概率中位数: {np.median(all_acceptance_probs):.6f}")
    
    print(f"\n各问题详细结果:")
    for i, r in enumerate(all_results, 1):
        print(f"  {i}. {r['question']}")
        print(f"     接受率: {r['acceptance_rate']:.2%}")
        print(f"     平均接受概率: {r['avg_acceptance_prob']:.6f}")
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

if __name__ == "__main__":
    main()

