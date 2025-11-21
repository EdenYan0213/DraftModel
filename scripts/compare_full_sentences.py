#!/usr/bin/env python3
"""
比较草稿模型和基础模型生成的完整句子相似度
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

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装，将使用简单的字符串相似度")

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
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        
        knowledge_config = config.get('knowledge_enhancement', {})
        use_vector_retrieval = knowledge_config.get('use_vector_retrieval', True)
        embedding_model_name = knowledge_config.get('embedding_model_name', None)
        
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=config['base_model']['hidden_size'],
            num_heads=config['base_model']['num_attention_heads'],
            cache_dim=config['knowledge_enhancement']['cache_dim'],
            use_vector_retrieval=use_vector_retrieval,
            embedding_model_name=embedding_model_name,
            target_model=target_model,  # 使用目标模型的嵌入层
            tokenizer=tokenizer  # 传入tokenizer
        )
        knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})
        
        if 'knowledge_embeddings' in cache_data:
            knowledge_cache_manager.knowledge_embeddings = cache_data['knowledge_embeddings']
        
        print(f"✓ 知识缓存加载完成，共 {len(knowledge_cache_manager.kv_cache)} 个知识项")
    
    # 创建草稿模型
    print("\n3. 创建草稿模型...")
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
    draft_model = draft_model.cpu()
    draft_model.eval()
    target_model.eval()
    
    # 加载训练好的权重（优先使用最新的best checkpoint）
    if checkpoint_path is None:
        checkpoint_dir = "output/checkpoints"
        # 优先查找best checkpoint，按epoch排序
        knowledge_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                                if 'knowledge' in f and 'best' in f and f.endswith('.pth')]
        if knowledge_checkpoints:
            # 按修改时间排序，最新的在前
            knowledge_checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            checkpoint_path = os.path.join(checkpoint_dir, knowledge_checkpoints[0])
            print(f"  自动选择最新的checkpoint: {knowledge_checkpoints[0]}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n4. 加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # 使用strict=False以兼容模型结构变化（如新增的归一化层）
        missing_keys, unexpected_keys = draft_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ 模型权重加载完成")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', 0):.4f}")
        if missing_keys:
            print(f"  ⚠ 缺失的键: {len(missing_keys)}个（可能是新增的层）")
        if unexpected_keys:
            print(f"  ⚠ 多余的键: {len(unexpected_keys)}个")
    else:
        print(f"\n⚠ 未找到训练好的模型，使用未训练的模型")
    
    return draft_model, target_model, tokenizer, knowledge_cache_manager

def generate_full_sentence(model, tokenizer, prompt: str, max_new_tokens: int = 50, 
                          knowledge_cache_manager=None, query_text=None, temperature: float = 0.7,
                          top_p: float = 0.9, use_top_p: bool = True, 
                          dynamic_repetition_penalty: bool = True):
    """生成完整句子（优化版）
    
    Args:
        temperature: 温度缩放参数，<1使分布更尖锐（提高确定性），>1使分布更平滑（增加多样性）
        top_p: nucleus采样参数，累积概率阈值（0.9表示保留累积概率90%的token）
        use_top_p: 是否使用top-p采样（True）或贪心解码（False）
        dynamic_repetition_penalty: 是否使用动态重复惩罚（根据已生成长度调整）
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    # 如果是草稿模型，需要传递知识缓存
    if isinstance(model, Qwen3DraftModel):
        # 检索知识（降低阈值以提高检索率，并获取相似度）
        knowledge_cache = None
        knowledge_similarity = None
        if knowledge_cache_manager is not None and query_text is not None:
            knowledge_config = model.config.get('knowledge_enhancement', {})
            threshold = knowledge_config.get('retrieval_threshold', 0.5)  # 从0.7降低到0.5
            retrieved = knowledge_cache_manager.retrieve(query_text, threshold=threshold, return_similarity=True)
            if retrieved is not None:
                if len(retrieved) == 4:
                    # 返回 (keys, values, similarity, answer_start_idx)
                    knowledge_cache = (retrieved[0], retrieved[1])
                    knowledge_similarity = retrieved[2]
                    answer_start_idx = retrieved[3]
                    # 设置模型的相似度、强制权重和答案起始位置
                    if hasattr(model, '_current_knowledge_similarity'):
                        model._current_knowledge_similarity = knowledge_similarity
                    if hasattr(model, '_current_answer_start_idx'):
                        model._current_answer_start_idx = answer_start_idx
                    if hasattr(model, '_force_knowledge_weight'):
                        if knowledge_similarity > 0.8:
                            model._force_knowledge_weight = 0.6  # 至少60%使用知识（更温和）
                        elif knowledge_similarity > 0.6:
                            model._force_knowledge_weight = 0.4  # 至少40%使用知识
                        else:
                            model._force_knowledge_weight = None  # 不强制，使用相似度调整
                elif len(retrieved) == 3:
                    # 可能是 (keys, values, similarity) 或 (keys, values, answer_start_idx)
                    if isinstance(retrieved[2], float):
                        knowledge_cache = (retrieved[0], retrieved[1])
                        knowledge_similarity = retrieved[2]
                        if hasattr(model, '_current_knowledge_similarity'):
                            model._current_knowledge_similarity = knowledge_similarity
                        if hasattr(model, '_current_answer_start_idx'):
                            model._current_answer_start_idx = None
                    else:
                        knowledge_cache = (retrieved[0], retrieved[1])
                        if hasattr(model, '_current_answer_start_idx'):
                            model._current_answer_start_idx = retrieved[2]
                else:
                    knowledge_cache = retrieved
        
        # 生成token（添加重复惩罚，特别处理换行符）
        current_input = input_ids
        generated = []
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        
        # 重复惩罚参数（动态调整）
        base_repetition_penalty = 1.2  # 基础重复惩罚
        no_repeat_ngram_size = 3  # 禁止3-gram重复（从2增加到3）
        newline_penalty = 3.0  # 对换行符的额外惩罚（增强）
        
        # 动态重复惩罚：根据已生成长度调整惩罚强度
        def get_repetition_penalty(generated_length: int) -> float:
            """动态调整重复惩罚：生成长度越长，惩罚越强"""
            if not dynamic_repetition_penalty:
                return base_repetition_penalty
            # 线性增加：每10个token增加0.05的惩罚
            penalty_increase = (generated_length // 10) * 0.05
            return min(base_repetition_penalty + penalty_increase, 1.5)  # 最大1.5
        
        # 识别换行符token ID
        newline_token_ids = set()
        for test_str in ['\n', '\r\n', '\r']:
            try:
                token_ids = tokenizer.encode(test_str, add_special_tokens=False)
                newline_token_ids.update(token_ids)
            except:
                pass
        
        # 用于跟踪已生成的token（用于重复惩罚）
        generated_token_ids = []
        consecutive_newlines = 0  # 连续换行符计数
        
        # 计算问题的长度（用于跟踪答案生成位置）
        question_length = input_ids.shape[1]
        
        # 设置模型的查询长度（用于动态mask）
        if hasattr(model, '_current_query_length'):
            model._current_query_length = question_length
        
        for i in range(max_new_tokens):
            # 计算当前已生成的答案长度（不包括问题部分）
            current_answer_length = current_input.shape[1] - question_length
            # 设置模型的当前答案长度（用于KV对齐）
            if hasattr(model, '_current_answer_length'):
                model._current_answer_length = current_answer_length
            
            outputs = model.forward(
                current_input, 
                knowledge_cache=knowledge_cache,
                retrieve_knowledge=False,
                query_text=query_text
            )
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 应用动态重复惩罚：对已生成的token降低概率
            current_repetition_penalty = get_repetition_penalty(len(generated_token_ids))
            if current_repetition_penalty > 1.0 and len(generated_token_ids) > 0:
                # 获取最近生成的token（用于惩罚，窗口大小也动态调整）
                window_size = min(15, len(generated_token_ids))  # 最多惩罚最近15个token
                recent_tokens = set(generated_token_ids[-window_size:])
                for token_id in recent_tokens:
                    if token_id < next_token_logits.size(-1):
                        # 如果logit是正数，除以penalty；如果是负数，乘以penalty
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= current_repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= current_repetition_penalty
            
            # 特别处理换行符：大幅降低换行符的概率（无论是否已经生成过）
            if newline_token_ids:
                for nl_id in newline_token_ids:
                    if nl_id < next_token_logits.size(-1):
                        # 对换行符应用更强的惩罚（基础惩罚 + 连续惩罚）
                        penalty = newline_penalty * (1 + consecutive_newlines * 2)
                        if next_token_logits[0, nl_id] > 0:
                            next_token_logits[0, nl_id] /= penalty
                        else:
                            next_token_logits[0, nl_id] *= penalty
            
            # 应用温度缩放
            scaled_logits = next_token_logits / temperature
            
            # 检查n-gram重复（防止短语重复）
            if no_repeat_ngram_size > 0 and len(generated_token_ids) >= no_repeat_ngram_size - 1:
                # 获取最近的n-1个token
                recent_ngram = generated_token_ids[-(no_repeat_ngram_size-1):]
                # 检查是否形成重复的n-gram
                for candidate_id in range(scaled_logits.size(-1)):
                    test_ngram = recent_ngram + [candidate_id]
                    # 检查这个n-gram是否在历史中出现过
                    if len(generated_token_ids) >= no_repeat_ngram_size:
                        for i in range(len(generated_token_ids) - no_repeat_ngram_size + 1):
                            if generated_token_ids[i:i+no_repeat_ngram_size] == test_ngram:
                                # 如果形成重复，大幅降低概率
                                scaled_logits[0, candidate_id] /= (current_repetition_penalty * 3)
                                break
            
            # Top-p (nucleus) 采样或贪心解码
            if use_top_p and top_p < 1.0:
                # 计算概率分布
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # 按概率排序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 找到累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # 保留第一个超过阈值的token（确保至少有一个token可选）
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                # 将超出范围的token概率设为0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                
                # 重新归一化
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # 从过滤后的分布中采样
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码
                next_token = torch.argmax(scaled_logits, dim=-1, keepdim=True)
            generated.append(next_token)
            token_id = next_token.item()
            generated_token_ids.append(token_id)
            
            # 检查是否是换行符
            is_newline = token_id in newline_token_ids
            if is_newline:
                consecutive_newlines += 1
                # 如果连续生成太多换行符（超过2个），强制停止
                if consecutive_newlines > 2:
                    break
            else:
                consecutive_newlines = 0
            
            current_input = torch.cat([current_input, next_token], dim=1)
            
            # 改进的停止条件
            if token_id == eos_token_id:
                break
        
        generated_ids = torch.cat(generated, dim=1) if generated else input_ids
        # 只解码新生成的部分
        new_tokens = generated_ids[0][len(input_ids[0]):]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        # 基础模型使用标准generate
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def compare_sentences(draft_model, target_model, tokenizer, question: str, 
                     max_new_tokens: int = 50, knowledge_cache_manager=None,
                     embedding_model=None, temperature: float = 0.7):
    """比较两个模型生成的句子"""
    print(f"\n{'='*70}")
    print(f"问题: {question}")
    print(f"{'='*70}")
    
    # 草稿模型生成（使用优化后的生成策略）
    print("\n📝 草稿模型生成中...")
    draft_text = generate_full_sentence(
        draft_model, tokenizer, question, 
        max_new_tokens=max_new_tokens,
        knowledge_cache_manager=knowledge_cache_manager,
        query_text=question,
        temperature=temperature,
        top_p=0.9,  # 使用top-p采样
        use_top_p=True,  # 启用top-p采样
        dynamic_repetition_penalty=True  # 启用动态重复惩罚
    )
    print(f"草稿模型输出: {draft_text}")
    
    # 基础模型生成
    print("\n📝 基础模型生成中...")
    target_text = generate_full_sentence(
        target_model, tokenizer, question,
        max_new_tokens=max_new_tokens
    )
    print(f"基础模型输出: {target_text}")
    
    # 计算相似度 - 优先使用embedding模型获取语义向量
    similarity = 0.0
    similarity_method = "计算失败"
    
    # 优先使用embedding模型（如果可用）
    if embedding_model is not None:
        try:
            # 使用embedding模型获取两句话的语义向量
            draft_embedding = embedding_model.encode(draft_text, convert_to_numpy=True)
            target_embedding = embedding_model.encode(target_text, convert_to_numpy=True)
            
            # 计算余弦相似度
            similarity = cosine_similarity(draft_embedding, target_embedding)
            similarity_method = "语义向量相似度（embedding模型）"
        except Exception as e:
            print(f"⚠ Embedding模型计算失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到目标模型的hidden states
            try:
                device = next(target_model.parameters()).device
                
                # 对草稿模型输出进行编码
                draft_inputs = tokenizer(draft_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                draft_input_ids = draft_inputs['input_ids'].to(device)
                
                # 对基础模型输出进行编码
                target_inputs = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                target_input_ids = target_inputs['input_ids'].to(device)
                
                with torch.no_grad():
                    # 获取草稿模型输出的hidden states（最后一层的输出）
                    draft_outputs = target_model(input_ids=draft_input_ids, output_hidden_states=True)
                    draft_hidden_states = draft_outputs.hidden_states[-1]  # 最后一层的hidden states
                    
                    # 获取基础模型输出的hidden states（最后一层的输出）
                    target_outputs = target_model(input_ids=target_input_ids, output_hidden_states=True)
                    target_hidden_states = target_outputs.hidden_states[-1]  # 最后一层的hidden states
                    
                    # 使用平均池化得到句子级表示（回退方案）
                    draft_embedding = draft_hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
                    target_embedding = target_hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
                    
                    similarity = cosine_similarity(draft_embedding, target_embedding)
                    similarity_method = "语义向量相似度（目标模型hidden states，回退）"
            except Exception as e2:
                print(f"⚠ 目标模型hidden states计算也失败: {e2}")
                similarity = 0.0
                similarity_method = "计算失败"
    else:
        # 如果没有embedding模型，回退到目标模型的hidden states
        try:
            device = next(target_model.parameters()).device
            
            # 对草稿模型输出进行编码
            draft_inputs = tokenizer(draft_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            draft_input_ids = draft_inputs['input_ids'].to(device)
            
            # 对基础模型输出进行编码
            target_inputs = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            target_input_ids = target_inputs['input_ids'].to(device)
            
            with torch.no_grad():
                # 获取草稿模型输出的hidden states（最后一层的输出）
                draft_outputs = target_model(input_ids=draft_input_ids, output_hidden_states=True)
                draft_hidden_states = draft_outputs.hidden_states[-1]  # 最后一层的hidden states
                
                # 获取基础模型输出的hidden states（最后一层的输出）
                target_outputs = target_model(input_ids=target_input_ids, output_hidden_states=True)
                target_hidden_states = target_outputs.hidden_states[-1]  # 最后一层的hidden states
                
                # 使用平均池化得到句子级表示
                draft_embedding = draft_hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
                target_embedding = target_hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
                
                similarity = cosine_similarity(draft_embedding, target_embedding)
                similarity_method = "语义向量相似度（目标模型hidden states，无embedding模型）"
        except Exception as e:
            print(f"⚠ 语义相似度计算失败: {e}")
            import traceback
            traceback.print_exc()
            similarity = 0.0
            similarity_method = "计算失败"
    
    print(f"\n📊 相似度分析:")
    print(f"  方法: {similarity_method}")
    print(f"  相似度: {similarity:.4f} ({similarity*100:.2f}%)")
    
    return {
        'question': question,
        'draft_text': draft_text,
        'target_text': target_text,
        'similarity': similarity,
        'similarity_method': similarity_method
    }

def main():
    """主函数"""
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 加载模型
    draft_model, target_model, tokenizer, knowledge_cache_manager = load_models(config_path)
    
    # 加载embedding模型用于相似度计算
    embedding_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n5. 加载embedding模型用于相似度计算...")
        try:
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✓ Embedding模型加载完成")
        except Exception as e:
            print(f"⚠ 无法加载embedding模型: {e}")
    
    # 测试问题（训练集:非训练集 = 4:1）
    # 训练集问题（来自知识库，8个，80%）
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
    
    # 非训练集问题（不在知识库中，2个，20%）
    non_training_questions = [
        "量子计算是",
        "边缘计算是",
    ]
    
    questions = training_questions + non_training_questions
    
    print("\n" + "="*70)
    print("完整句子生成与相似度比较")
    print("="*70)
    print(f"\n测试问题数量: {len(questions)}")
    print(f"  - 训练集问题: {len(training_questions)} 个 (80%)")
    print(f"  - 非训练集问题: {len(non_training_questions)} 个 (20%)")
    print(f"每个模型最多生成: 50 tokens")
    
    # 使用温度缩放（0.7使分布更尖锐，提高确定性）
    temperature = 0.7
    print(f"\n使用温度缩放: {temperature} (降低温度使分布更尖锐，提高接受率)")
    
    all_results = []
    
    for question in questions:
        result = compare_sentences(
            draft_model, target_model, tokenizer, question,
            max_new_tokens=50,
            knowledge_cache_manager=knowledge_cache_manager,
            embedding_model=embedding_model,
            temperature=temperature
        )
        all_results.append(result)
    
    # 总体汇总
    print("\n\n" + "="*70)
    print("总体汇总")
    print("="*70)
    
    similarities = [r['similarity'] for r in all_results]
    avg_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    
    print(f"\n平均相似度: {avg_similarity:.4f} ({avg_similarity*100:.2f}%)")
    print(f"相似度中位数: {median_similarity:.4f} ({median_similarity*100:.2f}%)")
    print(f"最高相似度: {max(similarities):.4f} ({max(similarities)*100:.2f}%)")
    print(f"最低相似度: {min(similarities):.4f} ({min(similarities)*100:.2f}%)")
    
    print(f"\n各问题详细结果:")
    print("\n【训练集问题】:")
    for i, r in enumerate(all_results[:len(training_questions)], 1):
        print(f"\n  {i}. {r['question']}")
        print(f"     草稿模型: {r['draft_text'][:80]}...")
        print(f"     基础模型: {r['target_text'][:80]}...")
        print(f"     相似度: {r['similarity']:.4f} ({r['similarity']*100:.2f}%)")
    
    print("\n【非训练集问题】:")
    for i, r in enumerate(all_results[len(training_questions):], 1):
        print(f"\n  {i}. {r['question']}")
        print(f"     草稿模型: {r['draft_text'][:80]}...")
        print(f"     基础模型: {r['target_text'][:80]}...")
        print(f"     相似度: {r['similarity']:.4f} ({r['similarity']*100:.2f}%)")
    
    # 分别统计训练集和非训练集的表现
    training_results = all_results[:len(training_questions)]
    non_training_results = all_results[len(training_questions):]
    
    training_similarities = [r['similarity'] for r in training_results]
    non_training_similarities = [r['similarity'] for r in non_training_results]
    
    print(f"\n【分类统计】:")
    print(f"  训练集问题平均相似度: {np.mean(training_similarities):.4f} ({np.mean(training_similarities)*100:.2f}%)")
    print(f"  非训练集问题平均相似度: {np.mean(non_training_similarities):.4f} ({np.mean(non_training_similarities)*100:.2f}%)")
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

if __name__ == "__main__":
    main()

