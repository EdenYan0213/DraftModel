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
                          knowledge_cache_manager=None, query_text=None, temperature: float = 0.7):
    """生成完整句子
    
    Args:
        temperature: 温度缩放参数，<1使分布更尖锐（提高确定性），>1使分布更平滑（增加多样性）
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    # 如果是草稿模型，需要传递知识缓存
    if isinstance(model, Qwen3DraftModel):
        # 检索知识
        knowledge_cache = None
        if knowledge_cache_manager is not None and query_text is not None:
            knowledge_config = model.config.get('knowledge_enhancement', {})
            threshold = knowledge_config.get('retrieval_threshold', 0.7)
            retrieved = knowledge_cache_manager.retrieve(query_text, threshold=threshold)
            if retrieved is not None:
                knowledge_cache = retrieved
        
        # 生成token（添加重复惩罚，特别处理换行符）
        current_input = input_ids
        generated = []
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        
        # 重复惩罚参数
        repetition_penalty = 1.2  # 对已生成的token降低概率（提高惩罚）
        no_repeat_ngram_size = 2  # 禁止2-gram重复
        newline_penalty = 3.0  # 对换行符的额外惩罚（增强）
        
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
        
        for i in range(max_new_tokens):
            outputs = model.forward(
                current_input, 
                knowledge_cache=knowledge_cache,
                retrieve_knowledge=False,
                query_text=query_text
            )
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 应用重复惩罚：对已生成的token降低概率
            if repetition_penalty > 1.0 and len(generated_token_ids) > 0:
                # 获取最近生成的token（用于惩罚）
                recent_tokens = set(generated_token_ids[-10:])  # 只惩罚最近10个token
                for token_id in recent_tokens:
                    if token_id < next_token_logits.size(-1):
                        # 如果logit是正数，除以penalty；如果是负数，乘以penalty
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= repetition_penalty
            
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
            
            # 检查2-gram重复（如果当前序列的最后1个token和候选token形成重复）
            if no_repeat_ngram_size > 0 and len(generated_token_ids) >= no_repeat_ngram_size - 1:
                # 获取最近的n-1个token
                recent_ngram = generated_token_ids[-(no_repeat_ngram_size-1):]
                # 如果下一个token和最近的token形成重复，降低其概率
                if len(recent_ngram) > 0 and recent_ngram[-1] < scaled_logits.size(-1):
                    # 对重复的token应用更强的惩罚
                    scaled_logits[0, recent_ngram[-1]] /= (repetition_penalty * 2)
            
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
            
            # 只检查EOS停止条件
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
    
    # 草稿模型生成
    print("\n📝 草稿模型生成中...")
    draft_text = generate_full_sentence(
        draft_model, tokenizer, question, 
        max_new_tokens=max_new_tokens,
        knowledge_cache_manager=knowledge_cache_manager,
        query_text=question,
        temperature=temperature
    )
    print(f"草稿模型输出: {draft_text}")
    
    # 基础模型生成
    print("\n📝 基础模型生成中...")
    target_text = generate_full_sentence(
        target_model, tokenizer, question,
        max_new_tokens=max_new_tokens
    )
    print(f"基础模型输出: {target_text}")
    
    # 计算相似度
    if embedding_model is not None:
        try:
            emb1 = embedding_model.encode(draft_text, convert_to_numpy=True)
            emb2 = embedding_model.encode(target_text, convert_to_numpy=True)
            similarity = cosine_similarity(emb1, emb2)
            similarity_method = "向量相似度（embedding）"
        except Exception as e:
            print(f"⚠ 向量相似度计算失败: {e}")
            similarity = 0.0
            similarity_method = "计算失败"
    else:
        similarity = 0.0
        similarity_method = "需要embedding模型"
    
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

