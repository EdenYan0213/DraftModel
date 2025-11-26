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
        ("深度学习是", "基于什么原理的？\n"+"深度学习是基于神经网络的原理，而神经网络是基于非线性映射的原理。具体来说，深度学习可以分为两种主要类型：卷积神经网络（CNNs）和循环神经网络"),
        ("自然语言处理是", "人工智能的分支，其目标是让计算机能够理解、分析和处理人类语言。人工智能的发展是基于自然语言处理的，而自然语言处理的发展是基于计算机视觉、语音识别和机器学习等技术。因此，人工智能的发展方向。"),
        ("计算机视觉是", "计算机视觉的学科，它研究图像和视频的处理，包括图像和视频的识别、分析、理解、处理和应用。计算机视觉是人工智能的重要组成部分，是人工智能的两大核心技术之一。"),
        ("强化学习是", "通过模拟环境来实现的，而机器学习是通过模拟环境来实现的。这说明了什么？\n"+"这是一个关于强化学习和机器学习的比较问题。强化学习和机器学习是两个不同的研究领域，它们都涉及模拟环境"),
        ("Transformer架构是", "基于什么架构？\n"+"Transformer架构是基于自注意力机制（Self-Attention）的。自注意力机制是Transformer模型的核心组件，它允许模型在处理输入序列时，动态地关注输入中的不同位置，从而实现更灵活和高效的计算。"),
        ("机器学习是", "人工智能的分支，它基于统计学和概率论，通过算法和数据训练来实现目标。人工智能的定义通常包括机器学习、深度学习、自然语言处理、计算机视觉等。"),
        ("注意力机制是", "神经网络中用于处理输入数据的机制，其核心在于将输入数据与输出数据进行某种方式的转换。在深度学习中，注意力机制被广泛应用于各种任务，如图像识别、自然语言处理等。"),
        ("神经网络是", "通过什么方式学习的？它如何处理输入和输出？它在处理输入时如何处理数据？它如何处理输出？它如何处理数据？它如何处理输入？它如何处理输出？"),
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

