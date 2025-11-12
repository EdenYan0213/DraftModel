#!/usr/bin/env python3
"""
构建知识缓存的脚本 - 从目标模型预计算常见问题的KV矩阵
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_cache import KnowledgeCacheManager

def main():
    """主函数"""
    print("=== 构建知识缓存 ===")
    
    # 加载配置
    config_path = "configs/qwen3_0.6b_config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化加载器
    loader = Qwen3Loader(config_path)
    
    # 加载目标模型和tokenizer
    print("\n1. 加载目标模型...")
    target_model = loader.load_target_model()
    tokenizer = loader.load_tokenizer()
    
    # 创建知识缓存管理器
    print("\n2. 创建知识缓存管理器...")
    base_config = config['base_model']
    knowledge_config = config.get('knowledge_enhancement', {})
    
    cache_manager = KnowledgeCacheManager(
        hidden_size=base_config['hidden_size'],
        num_heads=base_config['num_attention_heads'],
        cache_dim=knowledge_config.get('cache_dim', 512)
    )
    
    # 定义常见问题和答案（问题+答案的完整序列）
    # 格式: (问题, 答案) 或 (问题, None) 表示需要模型生成答案
    knowledge_items = [
        ("什么是深度学习？", "深度学习是机器学习的一个分支，通过多层神经网络学习数据的表示。"),
        ("什么是自然语言处理？", "自然语言处理是计算机科学和人工智能的一个分支，研究如何让计算机理解和生成人类语言。"),
        ("什么是Transformer架构？", "Transformer是一种基于注意力机制的神经网络架构，广泛应用于自然语言处理任务。"),
        ("什么是预训练语言模型？", "预训练语言模型是在大规模文本数据上预先训练的模型，可以用于各种下游任务。"),
        ("什么是强化学习？", "强化学习是机器学习的一个重要分支，通过与环境交互来学习最优策略。"),
        ("什么是计算机视觉？", "计算机视觉是人工智能的一个领域，研究如何让计算机理解和分析图像和视频。"),
        ("什么是大数据技术？", "大数据技术是处理和分析海量数据的技术和方法，包括存储、计算和分析。"),
        ("什么是云计算？", "云计算是通过互联网提供计算资源和服务的一种模式，包括存储、计算和应用服务。"),
        ("什么是区块链技术？", "区块链是一种分布式账本技术，通过密码学方法保证数据的安全性和不可篡改性。"),
        ("什么是物联网？", "物联网是将各种物理设备连接到互联网，实现设备之间的数据交换和智能控制。"),
        ("人工智能的发展历史", "人工智能的发展经历了符号主义、连接主义、深度学习等多个阶段，目前正处于快速发展期。"),
        ("机器学习的基本原理", "机器学习通过算法从数据中学习规律，建立模型来预测或分类新的数据。"),
        ("神经网络的工作原理", "神经网络通过多层神经元和权重连接，通过反向传播算法学习数据的特征表示。"),
        ("注意力机制的作用", "注意力机制允许模型在处理序列数据时，动态地关注不同位置的信息，提高模型的表达能力。"),
        ("BERT和GPT的区别", "BERT是双向编码器，适合理解任务；GPT是单向生成器，适合生成任务。"),
    ]
    
    print(f"\n3. 构建知识缓存（共 {len(knowledge_items)} 个知识项）...")
    print("   注意: 将缓存问题+答案的完整序列，特别是答案部分的KV")
    
    # 选择设备（优先CUDA，其次MPS，最后CPU）
    if torch.cuda.is_available():
        device = "cuda"
        target_model = target_model.cuda()
        cache_manager.kv_compressor = cache_manager.kv_compressor.cuda()
        cache_manager.kv_decompressor = cache_manager.kv_decompressor.cuda()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        target_model = target_model.to(device)
        cache_manager.kv_compressor = cache_manager.kv_compressor.to(device)
        cache_manager.kv_decompressor = cache_manager.kv_decompressor.to(device)
    else:
        device = "cpu"
        target_model = target_model.cpu()
        cache_manager.kv_compressor = cache_manager.kv_compressor.cpu()
        cache_manager.kv_decompressor = cache_manager.kv_decompressor.cpu()
    
    print(f"使用设备: {device}")
    
    # 从目标模型构建缓存
    # 使用hook机制提取真实的KV矩阵
    target_model.eval()
    
    # 存储提取的KV
    extracted_kvs = {}
    
    def extract_kv_hook(module, input, output):
        """Hook函数：提取attention层的KV"""
        # output通常是tuple: (attn_output, attn_weights, past_key_value)
        if isinstance(output, tuple) and len(output) >= 3:
            past_key_value = output[2]  # past_key_value
            if past_key_value is not None:
                k, v = past_key_value
                # k, v: (batch, num_heads, seq_len, head_dim)
                extracted_kvs['keys'] = k.squeeze(0)  # (num_heads, seq_len, head_dim)
                extracted_kvs['values'] = v.squeeze(0)
    
    with torch.no_grad():
        for i, (question, answer) in enumerate(knowledge_items):
            # 构建完整序列：问题 + 答案
            if answer is None:
                # 如果没有提供答案，使用模型生成
                print(f"  [{i+1}/{len(knowledge_items)}] 生成答案: '{question[:30]}...'")
                question_inputs = tokenizer(question, return_tensors="pt")
                question_ids = question_inputs['input_ids'].to(device)
                
                # 生成答案（贪心解码，最多50个token）
                generated_ids = question_ids.clone()
                for _ in range(50):
                    outputs = target_model(input_ids=generated_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                    
                    # 检查是否生成结束符
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                
                # 提取答案部分（去掉问题部分）
                answer_ids = generated_ids[:, question_ids.shape[1]:]
                full_sequence = generated_ids  # 完整序列：问题+答案
                answer_text = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
                print(f"    生成答案: {answer_text[:50]}...")
            else:
                # 使用提供的答案
                full_text = question + answer
                full_sequence = tokenizer(full_text, return_tensors="pt")['input_ids'].to(device)
                question_ids = tokenizer(question, return_tensors="pt")['input_ids'].to(device)
            
            # Tokenize完整序列（问题+答案）
            input_ids = full_sequence
            
            try:
                # 方法1: 尝试使用past_key_values
                outputs = target_model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=False
                )
                
                # 从past_key_values提取KV
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                    # 收集所有层的KV
                    all_keys = []
                    all_values = []
                    
                    for layer_idx, layer_kv in enumerate(outputs.past_key_values):
                        if layer_kv is not None:
                            k, v = layer_kv
                            # k, v: (batch, num_heads, seq_len, head_dim)
                            k_squeezed = k.squeeze(0)  # (num_heads, seq_len, head_dim)
                            v_squeezed = v.squeeze(0)
                            all_keys.append(k_squeezed)
                            all_values.append(v_squeezed)
                    
                    if all_keys:
                        # 平均所有层的KV（或者可以选择特定层）
                        avg_keys = torch.stack(all_keys).mean(dim=0)  # (num_heads, seq_len, head_dim)
                        avg_values = torch.stack(all_values).mean(dim=0)
                        
                        # 提取答案部分的KV（关键优化：只缓存答案部分的KV）
                        question_len = question_ids.shape[1]
                        answer_start_idx = question_len
                        
                        # 只取答案部分的KV（从问题结束位置到最后）
                        if answer_start_idx < avg_keys.shape[1]:
                            answer_keys = avg_keys[:, answer_start_idx:, :]  # (num_heads, answer_seq_len, head_dim)
                            answer_values = avg_values[:, answer_start_idx:, :]
                            
                            # 如果答案部分为空，使用完整序列的KV
                            if answer_keys.shape[1] == 0:
                                answer_keys = avg_keys
                                answer_values = avg_values
                                print(f"    警告: 答案部分为空，使用完整序列KV")
                        else:
                            # 如果问题长度超过序列长度，使用完整KV
                            answer_keys = avg_keys
                            answer_values = avg_values
                            print(f"    警告: 问题长度超过序列长度，使用完整KV")
                        
                        # 添加到缓存（使用问题作为key，但存储答案部分的KV）
                        cache_manager.add_knowledge(question, answer_keys, answer_values, compress=True)
                        print(f"  [{i+1}/{len(knowledge_items)}] '{question[:30]}...' - 完整序列KV: K={avg_keys.shape}, V={avg_values.shape}")
                        print(f"    答案部分KV: K={answer_keys.shape}, V={answer_values.shape} (已缓存)")
                    else:
                        print(f"  [{i+1}/{len(knowledge_items)}] '{question[:30]}...' - 警告: 未提取到KV")
                
                # 方法2: 如果没有past_key_values，通过hook提取
                elif hasattr(target_model, 'model') and hasattr(target_model.model, 'layers'):
                    # 注册hook到中间层（例如第14层，总共28层）
                    target_layer_idx = len(target_model.model.layers) // 2
                    target_layer = target_model.model.layers[target_layer_idx]
                    
                    if hasattr(target_layer, 'self_attn'):
                        hook_handle = target_layer.self_attn.register_forward_hook(extract_kv_hook)
                        
                        # 前向传播
                        _ = target_model(input_ids=input_ids)
                        
                        # 移除hook
                        hook_handle.remove()
                        
                        # 检查是否提取到KV
                        if 'keys' in extracted_kvs and 'values' in extracted_kvs:
                            k = extracted_kvs['keys']
                            v = extracted_kvs['values']
                            
                            # 提取答案部分的KV
                            question_len = question_ids.shape[1]
                            if question_len < k.shape[1]:
                                answer_keys = k[:, question_len:, :]
                                answer_values = v[:, question_len:, :]
                            else:
                                answer_keys = k
                                answer_values = v
                            
                            cache_manager.add_knowledge(question, answer_keys, answer_values, compress=True)
                            print(f"  [{i+1}/{len(knowledge_items)}] '{question[:30]}...' - 完整序列KV: K={k.shape}, V={v.shape}")
                            print(f"    答案部分KV: K={answer_keys.shape}, V={answer_values.shape} (已缓存)")
                        else:
                            print(f"  [{i+1}/{len(knowledge_items)}] '{question[:30]}...' - 警告: Hook未提取到KV")
                            # 使用fallback方法
                            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                            if hidden_states is not None:
                                seq_len = hidden_states.size(1)
                                num_heads = base_config['num_attention_heads']
                                head_dim = base_config['hidden_size'] // num_heads
                                k = hidden_states.squeeze(0).view(seq_len, num_heads, head_dim).transpose(0, 1)
                                v = k.clone()  # 简化处理
                                
                                # 提取答案部分的KV
                                question_len = question_ids.shape[1]
                                if question_len < k.shape[1]:
                                    answer_keys = k[:, question_len:, :]
                                    answer_values = v[:, question_len:, :]
                                else:
                                    answer_keys = k
                                    answer_values = v
                                
                                cache_manager.add_knowledge(question, answer_keys, answer_values, compress=True)
                                print(f"  [{i+1}/{len(knowledge_items)}] '{question[:30]}...' - 使用fallback方法")
                
                # 方法3: Fallback - 使用hidden states生成模拟KV
                else:
                    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                    if hidden_states is not None:
                        seq_len = hidden_states.size(1)
                        num_heads = base_config['num_attention_heads']
                        head_dim = base_config['hidden_size'] // num_heads
                        k = hidden_states.squeeze(0).view(seq_len, num_heads, head_dim).transpose(0, 1)
                        v = k.clone()  # 简化处理
                        
                        # 提取答案部分的KV
                        question_len = question_ids.shape[1]
                        if question_len < k.shape[1]:
                            answer_keys = k[:, question_len:, :]
                            answer_values = v[:, question_len:, :]
                        else:
                            answer_keys = k
                            answer_values = v
                        
                        cache_manager.add_knowledge(question, answer_keys, answer_values, compress=True)
                        print(f"  [{i+1}/{len(knowledge_items)}] '{question[:30]}...' - 使用fallback方法")
                    else:
                        print(f"  [{i+1}/{len(knowledge_items)}] '{question[:30]}...' - 错误: 无法提取KV")
                
                # 清空extracted_kvs
                extracted_kvs.clear()
                
            except Exception as e:
                print(f"  [{i+1}/{len(knowledge_items)}] 处理问题 '{question[:30]}...' 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n4. 保存知识缓存...")
    cache_dir = "output/knowledge_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 保存缓存
    cache_path = os.path.join(cache_dir, "knowledge_cache.pth")
    torch.save({
        'kv_cache': cache_manager.kv_cache,
        'compressed_cache': cache_manager.compressed_cache,
        'compressor_state': cache_manager.kv_compressor.state_dict(),
        'decompressor_state': cache_manager.kv_decompressor.state_dict(),
        'config': {
            'hidden_size': base_config['hidden_size'],
            'num_heads': base_config['num_attention_heads'],
            'cache_dim': knowledge_config.get('cache_dim', 512)
        }
    }, cache_path)
    
    print(f"知识缓存已保存到: {cache_path}")
    print(f"共缓存 {len(cache_manager)} 个知识项")
    
    # 测试检索
    print(f"\n5. 测试知识检索...")
    test_queries = ["深度学习", "自然语言处理", "机器学习"]
    for query in test_queries:
        kv = cache_manager.retrieve(query)
        if kv is not None:
            print(f"  查询 '{query}': 找到KV矩阵，形状 K={kv[0].shape}, V={kv[1].shape}")
        else:
            print(f"  查询 '{query}': 未找到匹配的缓存")
    
    print("\n=== 知识缓存构建完成 ===")

if __name__ == "__main__":
    main()

