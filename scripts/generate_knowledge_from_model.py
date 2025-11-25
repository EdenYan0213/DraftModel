#!/usr/bin/env python3
"""
使用基础模型生成问答对，构建知识缓存
"""

import os
import sys
import torch
import yaml
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_cache import KnowledgeCacheManager
from models.utils import get_device, print_device_info


def extract_prefill_vectors(target_model, tokenizer, question: str, answer: str, device: str = 'auto'):
    """提取prefill阶段的token向量"""
    # 确保设备参数被正确解析
    device = get_device(device)
    # 确保问题和答案之间有空格，以便更好的tokenization
    if not question.endswith(" ") and not answer.startswith(" "):
        full_text = question + " " + answer
    else:
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


def generate_answer(target_model, tokenizer, question: str, max_new_tokens: int = 100, 
                   temperature: float = 0.7, top_p: float = 0.9):
    """使用基础模型生成答案"""
    device = next(target_model.parameters()).device
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = target_model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
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
    if generated_text.startswith(question):
        generated_text = generated_text[len(question):].strip()
    
    return generated_text


def build_knowledge_cache_from_faq_pairs(faq_pairs, output_path=None, append=False, device: str = 'auto'):
    """从FAQ问答对构建知识缓存
    
    Args:
        faq_pairs: 问答对列表 [(question, answer), ...]
        output_path: 输出路径，如果为None则使用默认路径
        append: 是否追加到现有缓存（True）还是创建新缓存（False）
        device: 设备选择，'auto'自动选择，'cuda'使用CUDA，'cpu'使用CPU
    
    Returns:
        knowledge_cache_manager: 知识缓存管理器
    """
    config_path = "configs/qwen3_0.6b_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n1. 加载目标模型...")
    loader = Qwen3Loader(config_path)
    
    # 设备选择
    device = get_device(device)
    print_device_info(device)
    if device == 'cuda':
        print("  (加速知识缓存构建)")
    
    target_model = loader.load_target_model(device=device)
    tokenizer = loader.load_tokenizer()
    target_model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n2. 创建/加载知识缓存管理器...")
    cache_path = output_path or "output/knowledge_cache/knowledge_cache.pth"
    knowledge_cache_manager = KnowledgeCacheManager(
        hidden_size=config['base_model']['hidden_size'],
        use_vector_retrieval=True,
        embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2',
        target_model=target_model,
        tokenizer=tokenizer
    )
    
    if append and os.path.exists(cache_path):
        print(f"加载现有知识缓存: {cache_path}")
        knowledge_cache_manager.load(cache_path)
        print(f"现有知识项数量: {len(knowledge_cache_manager.knowledge_cache)}")
    else:
        print("创建新的知识缓存")
    
    print(f"\n3. 处理 {len(faq_pairs)} 条FAQ问答对...")
    print("="*70)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, (question, answer) in enumerate(faq_pairs, 1):
        print(f"\n[{i}/{len(faq_pairs)}] 问题: {question}")
        
        # 检查是否已存在
        if question in knowledge_cache_manager.knowledge_cache:
            print(f"  ⏭ 已存在，跳过")
            skip_count += 1
            continue
        
        try:
            print(f"  答案: {answer[:80]}..." if len(answer) > 80 else f"  答案: {answer}")
            
            # 提取向量
            print(f"  ⏳ 正在提取token向量...")
            token_vectors, answer_start_idx = extract_prefill_vectors(
                target_model, tokenizer, question, answer, device=device
            )
            
            # 添加到知识缓存
            knowledge_cache_manager.add_knowledge(
                key=question,
                question=question,
                answer=answer,
                token_vectors=token_vectors,
                answer_start_idx=answer_start_idx
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            error_count += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("生成统计")
    print("="*70)
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {error_count}")
    print(f"总计: {len(faq_pairs)}")
    print(f"当前知识项总数: {len(knowledge_cache_manager.knowledge_cache)}")
    
    return knowledge_cache_manager, success_count, skip_count, error_count


def main():
    """主函数 - 构建知识缓存（步骤1）"""
    print("="*70)
    print("构建知识缓存 (步骤1)")
    print("="*70)
    print("使用预定义的FAQ问答对构建知识缓存")
    print("="*70)
    
    print("\n准备FAQ问答对（多领域，30条）...")
    # 直接定义问答对，不调用模型生成
    faq_pairs = [
        # 领域1: 人工智能与机器学习 (6条)
        ("深度学习是", "人工智能的一个分支，它使用多层神经网络来学习数据的特征表示。深度学习通过模拟人脑神经网络的结构，能够自动从数据中提取复杂的模式和特征，在图像识别、自然语言处理、语音识别等领域取得了突破性进展。"),
        ("自然语言处理是", "计算机科学和人工智能的一个交叉领域，主要研究如何让计算机理解、处理和生成人类语言。自然语言处理包括文本分析、机器翻译、情感分析、问答系统、对话系统等多个子领域，是人工智能应用最广泛的领域之一。"),
        ("机器学习是", "人工智能的核心技术之一，它使计算机能够从数据中自动学习规律和模式，而无需明确编程。机器学习包括监督学习、无监督学习和强化学习三大类，广泛应用于推荐系统、搜索引擎、医疗诊断、金融风控等各个领域。"),
        ("什么是Transformer架构", "Transformer是一种基于注意力机制的神经网络架构，由Google在2017年提出。它完全摒弃了循环和卷积结构，仅使用自注意力机制来处理序列数据。Transformer架构在自然语言处理领域取得了革命性突破，成为BERT、GPT等现代大语言模型的基础架构。"),
        ("什么是注意力机制", "注意力机制是神经网络中用于解决长距离依赖问题的一种机制，它允许模型在处理序列时动态关注不同位置的信息。注意力机制通过计算查询、键和值之间的相似度，为每个位置分配不同的权重，从而让模型能够关注到最相关的信息。"),
        ("什么是神经网络", "神经网络是一种模拟人脑神经元结构的计算模型，它由多个层级的节点（神经元）组成，通过权重连接。神经网络能够通过训练自动学习输入和输出之间的复杂映射关系，是深度学习和机器学习的基础。"),
        
        # 领域2: 计算机科学 (6条)
        ("什么是算法", "算法是一系列解决问题的清晰指令，是计算机程序的核心。好的算法应该具有正确性、高效性、可读性和可维护性。算法设计包括排序、搜索、图算法、动态规划等多种方法，是计算机科学的基础。"),
        ("什么是数据结构", "数据结构是计算机存储和组织数据的方式，它决定了数据的访问方式和操作效率。常见的数据结构包括数组、链表、栈、队列、树、图、哈希表等。选择合适的数据结构可以大大提高程序的执行效率。"),
        ("什么是操作系统", "操作系统是管理计算机硬件和软件资源的系统软件，它是用户和计算机硬件之间的接口。操作系统负责进程管理、内存管理、文件系统管理、设备管理等核心功能，常见的操作系统包括Windows、Linux、macOS等。"),
        ("什么是数据库", "数据库是存储和管理数据的系统，它提供了高效的数据存储、查询、更新和管理功能。数据库系统包括关系型数据库（如MySQL、PostgreSQL）和非关系型数据库（如MongoDB、Redis），是信息系统的核心组件。"),
        ("什么是计算机网络", "计算机网络是将多台计算机通过通信线路连接起来，实现资源共享和信息传递的系统。计算机网络包括局域网、广域网、互联网等不同规模，采用TCP/IP协议栈进行通信，是现代信息社会的基础设施。"),
        ("什么是编程语言", "编程语言是用于编写计算机程序的形式化语言，它定义了程序员与计算机交流的语法和语义。编程语言包括低级语言（如汇编语言）和高级语言（如Python、Java、C++），不同的编程语言适用于不同的应用场景。"),
        
        # 领域3: 数学 (5条)
        ("什么是微积分", "微积分是数学的一个分支，主要研究函数的导数和积分。微积分包括微分学和积分学两部分，微分学研究函数的变化率，积分学研究函数的累积效应。微积分在物理学、工程学、经济学等领域有广泛应用。"),
        ("什么是线性代数", "线性代数是数学的一个分支，主要研究向量、矩阵、线性方程组等概念。线性代数在计算机图形学、机器学习、信号处理等领域有重要应用，是深度学习和数据科学的基础数学工具。"),
        ("什么是概率论", "概率论是数学的一个分支，研究随机现象的规律性。概率论包括概率、随机变量、概率分布、期望、方差等概念，在统计学、机器学习、金融风险管理等领域有广泛应用。"),
        ("什么是统计学", "统计学是收集、分析、解释和呈现数据的科学。统计学包括描述统计和推断统计，描述统计用于总结数据特征，推断统计用于从样本推断总体特征。统计学是数据科学和机器学习的重要基础。"),
        ("什么是优化理论", "优化理论是数学的一个分支，研究在给定约束条件下寻找最优解的方法。优化理论包括线性规划、非线性规划、凸优化等方法，在机器学习、运筹学、工程设计中都有重要应用。"),
        
        # 领域4: 物理学 (4条)
        ("什么是量子力学", "量子力学是描述微观粒子行为的物理学理论，它是现代物理学的两大支柱之一。量子力学揭示了微观世界的概率性和不确定性，在原子物理、凝聚态物理、量子计算等领域有重要应用。"),
        ("什么是相对论", "相对论是爱因斯坦提出的物理学理论，包括狭义相对论和广义相对论。狭义相对论研究高速运动物体的时空关系，广义相对论研究引力的本质。相对论改变了我们对时空和引力的理解。"),
        ("什么是电磁学", "电磁学是研究电和磁现象及其相互关系的物理学分支。电磁学包括静电学、电流、磁场、电磁感应等内容，是理解现代电子技术和通信技术的基础。"),
        ("什么是热力学", "热力学是研究热现象和能量转换的物理学分支。热力学包括热力学定律、熵、热机效率等概念，在工程热力学、化学热力学、统计力学等领域有重要应用。"),
        
        # 领域5: 生物学 (4条)
        ("什么是DNA", "DNA（脱氧核糖核酸）是生物体遗传信息的载体，它由四种碱基（A、T、G、C）组成双螺旋结构。DNA通过复制传递遗传信息，通过转录和翻译指导蛋白质合成，是生命的基础分子。"),
        ("什么是细胞", "细胞是生物体的基本结构和功能单位，所有生物体都由一个或多个细胞组成。细胞包括原核细胞和真核细胞，具有细胞膜、细胞质、细胞核等结构，是生命活动的基本单位。"),
        ("什么是进化论", "进化论是解释生物多样性和适应性的科学理论，由达尔文提出。进化论认为生物通过自然选择、遗传变异等机制不断进化，适者生存，不适者被淘汰。进化论是现代生物学的核心理论。"),
        ("什么是基因", "基因是遗传信息的基本单位，是DNA分子上具有特定功能的片段。基因通过编码蛋白质或RNA来影响生物体的性状，基因的变异是生物进化和遗传疾病的基础。"),
        
        # 领域6: 历史与文化 (3条)
        ("什么是文艺复兴", "文艺复兴是14-17世纪欧洲的一场思想文化运动，它标志着中世纪的结束和现代文明的开始。文艺复兴强调人文主义，重视人的价值和创造力，在文学、艺术、科学等领域产生了深远影响。"),
        ("什么是工业革命", "工业革命是18-19世纪从英国开始的重大历史变革，它标志着人类社会从农业文明向工业文明的转变。工业革命通过机械化、工厂化生产，极大地提高了生产力，改变了人类的生活方式和社会结构。"),
        ("什么是古代文明", "古代文明是人类早期发展的重要阶段，包括古埃及、古希腊、古罗马、古中国、古印度等文明。这些文明在政治、经济、文化、科技等方面取得了辉煌成就，为现代文明奠定了基础。"),
        
        # 领域7: 经济学 (2条)
        ("什么是市场经济", "市场经济是一种以市场机制配置资源的经济体制，它通过价格信号引导资源的流动和配置。市场经济强调自由竞争、私有产权、价格机制，是现代社会主要的经济组织形式。"),
        ("什么是通货膨胀", "通货膨胀是指货币供应量增加导致物价普遍上涨的经济现象。适度的通货膨胀有利于经济发展，但过度的通货膨胀会降低货币购买力，影响经济稳定和人民生活水平。"),
    ]
    
    print(f"共准备 {len(faq_pairs)} 条FAQ问答对，涵盖7个领域")
    
    # 构建知识缓存
    # 注意：如果append=True，会保留旧的知识项（可能没有qa_pairs）
    # 如果append=False，会创建全新的缓存，确保一致性
    knowledge_cache_manager, success_count, skip_count, error_count = build_knowledge_cache_from_faq_pairs(
        faq_pairs=faq_pairs,
        output_path=None,  # 使用默认路径
        append=False  # 创建新缓存，确保所有知识项都有qa_pairs
    )
    
    print("\n保存知识缓存...")
    output_dir = Path("output/knowledge_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "knowledge_cache.pth"
    knowledge_cache_manager.save(str(cache_path))
    
    print("\n" + "="*70)
    print("知识缓存构建完成！")
    print("="*70)
    print(f"缓存文件: {cache_path}")
    print(f"知识项数量: {len(knowledge_cache_manager.knowledge_cache)}")
    print("\n" + "="*70)
    print("下一步操作")
    print("="*70)
    print("知识缓存已构建完成，现在可以训练模型：")
    print("  python scripts/train.py")
    print("\n或者先测试推理（使用已有模型）：")
    print("  python scripts/inference.py")


if __name__ == "__main__":
    main()

