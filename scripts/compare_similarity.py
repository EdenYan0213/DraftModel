#!/usr/bin/env python3
"""
比较草稿模型和基础模型生成答案的语义相似度
"""

import os
import sys
import torch
import yaml
from pathlib import Path
import time
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.draft_model import Qwen3DraftModel
from models.knowledge_cache import KnowledgeCacheManager

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("错误: 需要安装 sentence-transformers")
    print("运行: pip install sentence-transformers")
    sys.exit(1)


def load_model(config_path: str, checkpoint_path: str, device: str = 'cpu'):
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
    return draft_model, target_model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 0.3, top_p: float = 0.9):
    """生成文本"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 检查是否是草稿模型（有retrieve_knowledge参数）
            if hasattr(model, 'knowledge_cache_manager'):
                outputs = model(
                    input_ids=generated_ids,
                    retrieve_knowledge=True,
                    query_text=prompt
                )
            else:
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


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def compute_semantic_similarity(text1: str, text2: str, embedding_model) -> float:
    """计算两个文本的语义相似度"""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    try:
        embeddings = embedding_model.encode([text1, text2], convert_to_numpy=True)
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        return float(similarity)
    except Exception as e:
        print(f"⚠ 计算相似度时出错: {e}")
        return 0.0


def main():
    """主函数"""
    # 设置随机种子确保可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    print("="*70)
    print("草稿模型 vs 基础模型 - 语义相似度比较")
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
        print("错误: 未找到检查点文件")
        return
    
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"使用检查点: {checkpoint_path}")
    
    # 设备选择
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用设备: CUDA")
    else:
        print(f"使用设备: CPU")
    
    # 加载模型
    draft_model, target_model, tokenizer = load_model(
        config_path, checkpoint_path, device=device
    )
    
    # 加载嵌入模型
    print("\n加载嵌入模型用于相似度计算...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✓ 嵌入模型加载完成")
    
    # 测试提示
    test_prompts = [
        "深度学习是",
        "自然语言处理是",
        "Transformer架构是",
        "机器学习是",
        "注意力机制是",
        "神经网络是",
        "计算机视觉是",
        "强化学习是",
    ]
    
    print("\n" + "="*70)
    print("生成文本并计算相似度")
    print("="*70)
    
    similarities = []
    generation_times_draft = []
    generation_times_target = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(test_prompts)}: {prompt}")
        print(f"{'='*70}")
        
        # 草稿模型生成
        print(f"\n[草稿模型生成]")
        start_time = time.time()
        draft_text = generate_text(draft_model, tokenizer, prompt, max_new_tokens=50)
        draft_time = time.time() - start_time
        generation_times_draft.append(draft_time)
        print(f"生成时间: {draft_time:.2f}秒")
        print(f"生成文本: {draft_text[:200]}..." if len(draft_text) > 200 else f"生成文本: {draft_text}")
        
        # 目标模型生成
        print(f"\n[基础模型生成]")
        start_time = time.time()
        target_text = generate_text(target_model, tokenizer, prompt, max_new_tokens=50)
        target_time = time.time() - start_time
        generation_times_target.append(target_time)
        print(f"生成时间: {target_time:.2f}秒")
        print(f"生成文本: {target_text[:200]}..." if len(target_text) > 200 else f"生成文本: {target_text}")
        
        # 计算语义相似度
        similarity = compute_semantic_similarity(draft_text, target_text, embedding_model)
        similarities.append(similarity)
        
        print(f"\n[相似度分析]")
        print(f"语义相似度: {similarity:.4f} ({similarity*100:.2f}%)")
        
        if similarity >= 0.8:
            print("✓ 相似度很高，生成内容非常相似")
        elif similarity >= 0.6:
            print("✓ 相似度较高，生成内容较为相似")
        elif similarity >= 0.4:
            print("⚠ 相似度中等，生成内容有一定差异")
        else:
            print("✗ 相似度较低，生成内容差异较大")
        
        # 速度对比
        if draft_time > 0:
            speedup = target_time / draft_time
            print(f"速度提升: {speedup:.2f}x")
    
    # 统计结果
    print("\n" + "="*70)
    print("统计结果")
    print("="*70)
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    print(f"\n语义相似度统计:")
    print(f"  平均相似度: {avg_similarity:.4f} ({avg_similarity*100:.2f}%)")
    print(f"  标准差: {std_similarity:.4f}")
    print(f"  最低相似度: {min_similarity:.4f} ({min_similarity*100:.2f}%)")
    print(f"  最高相似度: {max_similarity:.4f} ({max_similarity*100:.2f}%)")
    
    avg_speedup = np.mean([t2/t1 for t1, t2 in zip(generation_times_draft, generation_times_target) if t1 > 0])
    print(f"\n速度统计:")
    print(f"  平均速度提升: {avg_speedup:.2f}x")
    print(f"  草稿模型平均生成时间: {np.mean(generation_times_draft):.2f}秒")
    print(f"  基础模型平均生成时间: {np.mean(generation_times_target):.2f}秒")
    
    # 详细相似度列表
    print(f"\n各测试用例相似度:")
    for i, (prompt, sim) in enumerate(zip(test_prompts, similarities), 1):
        print(f"  {i}. {prompt[:20]:<20} : {sim:.4f} ({sim*100:.2f}%)")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

