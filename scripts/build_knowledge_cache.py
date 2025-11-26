#!/usr/bin/env python3
"""
构建知识缓存 - 存储prefill阶段的token向量
"""

import os
import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_cache import KnowledgeCacheManager
from models.utils import get_device, print_device_info


def extract_prefill_vectors(target_model, tokenizer, question: str, answer: str, device: str = 'auto'):
    """提取prefill阶段的token向量"""
    # 确保设备参数被正确解析
    device = get_device(device)
    full_text = question + answer
    
    inputs = tokenizer(full_text, return_tensors="pt", padding=False, truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    
    question_inputs = tokenizer(question, return_tensors="pt", padding=False, truncation=True, max_length=2048)
    question_len = question_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = target_model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        token_vectors = hidden_states.squeeze(0)
        # 移动到CPU以便保存（保存时统一在CPU上，加载时更灵活）
        token_vectors = token_vectors.cpu()
        answer_start_idx = question_len
    
    return token_vectors, answer_start_idx


def main():
    """主函数"""
    print("="*70)
    print("构建知识缓存（prefill阶段token向量）")
    print("="*70)
    
    config_path = "configs/qwen3_0.6b_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n1. 加载目标模型...")
    loader = Qwen3Loader(config_path)
    
    # 自动选择设备：优先CUDA，其次CPU
    device = get_device('auto')
    print_device_info(device)
    if device == 'cuda':
        print("  (加速知识缓存构建)")
    
    target_model = loader.load_target_model(device=device)
    tokenizer = loader.load_tokenizer()
    target_model.eval()
    
    print("\n2. 创建知识缓存管理器...")
    knowledge_cache_manager = KnowledgeCacheManager(
        hidden_size=config['base_model']['hidden_size'],
        use_vector_retrieval=True,
        embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2',
        target_model=target_model,
        tokenizer=tokenizer
    )
    
    print("\n3. 构建知识缓存...")
    knowledge_base = [
        ("深度学习是", "人工智能的一个分支，它使用多层神经网络来学习数据的特征表示。"),
        ("自然语言处理是", "人工智能的一个分支，它主要研究如何让计算机理解并处理人类语言。"),
        ("计算机视觉是", "计算机科学与工程的交叉学科，它涉及到图像处理、模式识别、机器学习和人工智能等技术。"),
        ("强化学习是", "人工智能的一个重要分支，它在很多领域都有广泛的应用，包括自动驾驶、机器人、医疗诊断等。"),
        ("Transformer架构是", "一种基于注意力机制的神经网络架构，它在自然语言处理领域取得了突破性进展。"),
        ("机器学习是", "人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。"),
        ("注意力机制是", "神经网络中用于解决长距离依赖问题的一种机制，它允许模型在处理序列时关注不同位置的信息。"),
        ("神经网络是", "一种模拟人脑神经元结构的计算模型，它由多个层级的节点（神经元）组成，通过权重连接。"),
    ]
    
    for question, answer in knowledge_base:
        print(f"\n处理: {question}")
        try:
            token_vectors, answer_start_idx = extract_prefill_vectors(
                target_model, tokenizer, question, answer, device=device
            )
            knowledge_cache_manager.add_knowledge(
                key=question,
                question=question,
                answer=answer,
                token_vectors=token_vectors,
                answer_start_idx=answer_start_idx
            )
        except Exception as e:
            print(f"⚠ 处理失败: {e}")
    
    print("\n4. 保存知识缓存...")
    output_dir = Path("output/knowledge_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "knowledge_cache.pth"
    knowledge_cache_manager.save(str(cache_path))
    
    print("\n" + "="*70)
    print("知识缓存构建完成！")
    print("="*70)
    print(f"缓存文件: {cache_path}")
    print(f"知识项数量: {len(knowledge_cache_manager.knowledge_cache)}")


if __name__ == "__main__":
    main()

