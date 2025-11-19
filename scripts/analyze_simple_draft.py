#!/usr/bin/env python3
"""
简单草稿模型推理分析 - 对比草稿模型和基础模型
"""

import os
import sys
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.simple_draft_model import SimpleDraftModel

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装，无法计算句子相似度")


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
    
    print(f"✓ 基础模型加载完成，总层数: {len(target_model.model.layers)}")
    
    # 创建简单草稿模型
    print("\n2. 创建简单草稿模型...")
    simple_draft = SimpleDraftModel(config, target_model)
    simple_draft = simple_draft.cpu()
    simple_draft.eval()
    target_model.eval()
    
    # 加载训练好的权重
    if checkpoint_path is None:
        checkpoint_dir = "output/checkpoints/simple_draft"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                # 优先使用best模型
                best_checkpoints = [f for f in checkpoints if 'best' in f]
                if best_checkpoints:
                    best_checkpoints.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
                    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoints[0])
                else:
                    checkpoints.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n3. 加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        simple_draft.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型权重加载完成")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', 0):.4f}")
    else:
        print(f"\n⚠ 未找到训练好的模型，使用未训练的模型")
    
    return simple_draft, target_model, tokenizer


def analyze_token_acceptance(draft_model, target_model, tokenizer, question: str, max_new_tokens: int = 5):
    """分析token级接受率"""
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
            draft_outputs = draft_model(current_input, attention_mask=attention_mask)
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
    
    # 汇总
    acceptance_rate = sum(results['is_accepted']) / len(results['is_accepted']) * 100
    avg_acceptance_prob = np.mean(results['acceptance_probs'])
    
    print(f"\n{'='*70}")
    print("汇总")
    print(f"{'='*70}")
    print(f"接受率: {acceptance_rate:.2f}% ({sum(results['is_accepted'])}/{len(results['is_accepted'])})")
    print(f"平均接受概率: {avg_acceptance_prob:.6f}")
    
    results['acceptance_rate'] = acceptance_rate
    results['avg_acceptance_prob'] = avg_acceptance_prob
    
    return results


def generate_full_sentence(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """生成完整句子"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    # 检查是否是基础模型（有generate方法）
    if hasattr(model, 'generate') and not hasattr(model, 'cross_attentions'):
        # 基础模型使用标准的generate方法
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 添加重复惩罚，防止重复生成
                no_repeat_ngram_size=2   # 禁止2-gram重复
            )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
    else:
        # 草稿模型使用手动循环生成
        current_input = input_ids
        generated = []
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = model(current_input)
                # 处理不同的输出格式
                if isinstance(outputs, dict):
                    next_token_logits = outputs['logits'][:, -1, :]
                else:
                    next_token_logits = outputs.logits[:, -1, :]
                
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated.append(next_token)
                current_input = torch.cat([current_input, next_token], dim=1)
                
                token_id = next_token.item()
                if token_id == eos_token_id:
                    break
        
        generated_ids = torch.cat(generated, dim=1) if generated else input_ids
        new_tokens = generated_ids[0][len(input_ids[0]):]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def main():
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 加载模型
    draft_model, target_model, tokenizer = load_models(config_path)
    
    # 加载embedding模型用于相似度计算
    embedding_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n3. 加载embedding模型用于相似度计算...")
        try:
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✓ Embedding模型加载完成")
        except Exception as e:
            print(f"⚠ 无法加载embedding模型: {e}")
    
    # 测试问题
    training_questions = [
        "深度学习是",
        "自然语言处理是",
        "计算机视觉是",
        "强化学习是",
        "Transformer架构是",
        "机器学习是",
        "注意力机制是",
        "神经网络是",
    ]
    
    non_training_questions = [
        "量子计算是",
        "边缘计算是",
    ]
    
    questions = training_questions + non_training_questions
    
    print("\n" + "="*70)
    print("Token级接受率分析")
    print("="*70)
    print(f"\n测试问题数量: {len(questions)}")
    print(f"  - 训练集问题: {len(training_questions)} 个")
    print(f"  - 非训练集问题: {len(non_training_questions)} 个")
    
    all_results = []
    
    for question in questions:
        results = analyze_token_acceptance(draft_model, target_model, tokenizer, question, max_new_tokens=5)
        all_results.append(results)
    
    # 总体汇总
    print("\n\n" + "="*70)
    print("总体汇总")
    print("="*70)
    
    total_accepted = sum(sum(r['is_accepted']) for r in all_results)
    total_tokens = sum(len(r['is_accepted']) for r in all_results)
    overall_acceptance_rate = total_accepted / total_tokens * 100 if total_tokens > 0 else 0
    avg_acceptance_prob = np.mean([np.mean(r['acceptance_probs']) for r in all_results])
    
    print(f"\n总体接受率: {overall_acceptance_rate:.2f}% ({total_accepted}/{total_tokens})")
    print(f"平均接受概率: {avg_acceptance_prob:.6f}")
    
    # 分类统计
    training_results = all_results[:len(training_questions)]
    non_training_results = all_results[len(training_questions):]
    
    training_accepted = sum(sum(r['is_accepted']) for r in training_results)
    training_total = sum(len(r['is_accepted']) for r in training_results)
    training_rate = training_accepted / training_total * 100 if training_total > 0 else 0
    
    non_training_accepted = sum(sum(r['is_accepted']) for r in non_training_results)
    non_training_total = sum(len(r['is_accepted']) for r in non_training_results)
    non_training_rate = non_training_accepted / non_training_total * 100 if non_training_total > 0 else 0
    
    print(f"\n【分类统计】:")
    print(f"  训练集问题接受率: {training_rate:.2f}% ({training_accepted}/{training_total})")
    print(f"  非训练集问题接受率: {non_training_rate:.2f}% ({non_training_accepted}/{non_training_total})")
    
    # 完整句子相似度分析
    if embedding_model is not None:
        print("\n\n" + "="*70)
        print("完整句子相似度分析")
        print("="*70)
        
        sentence_results = []
        
        for question in questions[:5]:  # 只测试前5个问题
            print(f"\n问题: {question}")
            
            draft_text = generate_full_sentence(draft_model, tokenizer, question, max_new_tokens=50)
            target_text = generate_full_sentence(target_model, tokenizer, question, max_new_tokens=50)
            
            print(f"草稿模型: {draft_text[:80]}...")
            print(f"基础模型: {target_text[:80]}...")
            
            similarity = 0.0
            if draft_text and target_text:
                try:
                    draft_embedding = embedding_model.encode(draft_text, convert_to_numpy=True)
                    target_embedding = embedding_model.encode(target_text, convert_to_numpy=True)
                    similarity = cosine_similarity(draft_embedding, target_embedding)
                except Exception as e:
                    print(f"⚠ 计算相似度失败: {e}")
            
            print(f"相似度: {similarity:.4f} ({similarity*100:.2f}%)")
            sentence_results.append(similarity)
        
        if sentence_results:
            avg_similarity = np.mean(sentence_results)
            print(f"\n平均相似度: {avg_similarity:.4f} ({avg_similarity*100:.2f}%)")
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)


if __name__ == "__main__":
    main()

