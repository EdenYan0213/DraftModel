#!/usr/bin/env python3
"""
构建多个知识缓存文件
支持按领域或用途分类构建不同的知识缓存
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接导入构建函数
from generate_knowledge_from_model import build_knowledge_cache_from_faq_pairs


def main():
    """主函数 - 构建多个知识缓存"""
    print("="*70)
    print("构建多个知识缓存")
    print("="*70)
    
    # 定义不同领域的FAQ对
    ai_ml_faqs = [
        # 人工智能与机器学习
        ("深度学习是", "人工智能的一个分支，它使用多层神经网络来学习数据的特征表示。深度学习通过模拟人脑神经网络的结构，能够自动从数据中提取复杂的模式和特征，在图像识别、自然语言处理、语音识别等领域取得了突破性进展。"),
        ("自然语言处理是", "计算机科学和人工智能的一个交叉领域，主要研究如何让计算机理解、处理和生成人类语言。自然语言处理包括文本分析、机器翻译、情感分析、问答系统、对话系统等多个子领域，是人工智能应用最广泛的领域之一。"),
        ("机器学习是", "人工智能的核心技术之一，它使计算机能够从数据中自动学习规律和模式，而无需明确编程。机器学习包括监督学习、无监督学习和强化学习三大类，广泛应用于推荐系统、搜索引擎、医疗诊断、金融风控等各个领域。"),
        ("什么是Transformer架构", "Transformer是一种基于注意力机制的神经网络架构，由Google在2017年提出。它完全摒弃了循环和卷积结构，仅使用自注意力机制来处理序列数据。Transformer架构在自然语言处理领域取得了革命性突破，成为BERT、GPT等现代大语言模型的基础架构。"),
        ("什么是注意力机制", "注意力机制是神经网络中用于解决长距离依赖问题的一种机制，它允许模型在处理序列时动态关注不同位置的信息。注意力机制通过计算查询、键和值之间的相似度，为每个位置分配不同的权重，从而让模型能够关注到最相关的信息。"),
        ("什么是神经网络", "神经网络是一种模拟人脑神经元结构的计算模型，它由多个层级的节点（神经元）组成，通过权重连接。神经网络能够通过训练自动学习输入和输出之间的复杂映射关系，是深度学习和机器学习的基础。"),
    ]
    
    cs_faqs = [
        # 计算机科学
        ("什么是算法", "算法是一系列解决问题的清晰指令，是计算机程序的核心。好的算法应该具有正确性、高效性、可读性和可维护性。算法设计包括排序、搜索、图算法、动态规划等多种方法，是计算机科学的基础。"),
        ("什么是数据结构", "数据结构是计算机存储和组织数据的方式，它决定了数据的访问方式和操作效率。常见的数据结构包括数组、链表、栈、队列、树、图、哈希表等。选择合适的数据结构可以大大提高程序的执行效率。"),
        ("什么是操作系统", "操作系统是管理计算机硬件和软件资源的系统软件，它是用户和计算机硬件之间的接口。操作系统负责进程管理、内存管理、文件系统管理、设备管理等核心功能，常见的操作系统包括Windows、Linux、macOS等。"),
        ("什么是数据库", "数据库是存储和管理数据的系统，它提供了高效的数据存储、查询、更新和管理功能。数据库系统包括关系型数据库（如MySQL、PostgreSQL）和非关系型数据库（如MongoDB、Redis），是信息系统的核心组件。"),
        ("什么是计算机网络", "计算机网络是将多台计算机通过通信线路连接起来，实现资源共享和信息传递的系统。计算机网络包括局域网、广域网、互联网等不同规模，采用TCP/IP协议栈进行通信，是现代信息社会的基础设施。"),
        ("什么是编程语言", "编程语言是用于编写计算机程序的形式化语言，它定义了程序员与计算机交流的语法和语义。编程语言包括低级语言（如汇编语言）和高级语言（如Python、Java、C++），不同的编程语言适用于不同的应用场景。"),
    ]
    
    math_faqs = [
        # 数学
        ("什么是微积分", "微积分是数学的一个分支，主要研究函数的导数和积分。微积分包括微分学和积分学两部分，微分学研究函数的变化率，积分学研究函数的累积效应。微积分在物理学、工程学、经济学等领域有广泛应用。"),
        ("什么是线性代数", "线性代数是数学的一个分支，主要研究向量、矩阵、线性方程组等概念。线性代数在计算机图形学、机器学习、信号处理等领域有重要应用，是深度学习和数据科学的基础数学工具。"),
        ("什么是概率论", "概率论是数学的一个分支，研究随机现象的规律性。概率论包括概率、随机变量、概率分布、期望、方差等概念，在统计学、机器学习、金融风险管理等领域有广泛应用。"),
        ("什么是统计学", "统计学是收集、分析、解释和呈现数据的科学。统计学包括描述统计和推断统计，描述统计用于总结数据特征，推断统计用于从样本推断总体特征。统计学是数据科学和机器学习的重要基础。"),
        ("什么是优化理论", "优化理论是数学的一个分支，研究在给定约束条件下寻找最优解的方法。优化理论包括线性规划、非线性规划、凸优化等方法，在机器学习、运筹学、工程设计中都有重要应用。"),
    ]
    
    physics_faqs = [
        # 物理学
        ("什么是量子力学", "量子力学是描述微观粒子行为的物理学理论，它是现代物理学的两大支柱之一。量子力学揭示了微观世界的概率性和不确定性，在原子物理、凝聚态物理、量子计算等领域有重要应用。"),
        ("什么是相对论", "相对论是爱因斯坦提出的物理学理论，包括狭义相对论和广义相对论。狭义相对论研究高速运动物体的时空关系，广义相对论研究引力的本质。相对论改变了我们对时空和引力的理解。"),
        ("什么是电磁学", "电磁学是研究电和磁现象及其相互关系的物理学分支。电磁学包括静电学、电流、磁场、电磁感应等内容，是理解现代电子技术和通信技术的基础。"),
        ("什么是热力学", "热力学是研究热现象和能量转换的物理学分支。热力学包括热力学定律、熵、热机效率等概念，在工程热力学、化学热力学、统计力学等领域有重要应用。"),
    ]
    
    biology_faqs = [
        # 生物学
        ("什么是DNA", "DNA（脱氧核糖核酸）是生物体遗传信息的载体，它由四种碱基（A、T、G、C）组成双螺旋结构。DNA通过复制传递遗传信息，通过转录和翻译指导蛋白质合成，是生命的基础分子。"),
        ("什么是细胞", "细胞是生物体的基本结构和功能单位，所有生物体都由一个或多个细胞组成。细胞包括原核细胞和真核细胞，具有细胞膜、细胞质、细胞核等结构，是生命活动的基本单位。"),
        ("什么是进化论", "进化论是解释生物多样性和适应性的科学理论，由达尔文提出。进化论认为生物通过自然选择、遗传变异等机制不断进化，适者生存，不适者被淘汰。进化论是现代生物学的核心理论。"),
        ("什么是基因", "基因是遗传信息的基本单位，是DNA分子上具有特定功能的片段。基因通过编码蛋白质或RNA来影响生物体的性状，基因的变异是生物进化和遗传疾病的基础。"),
    ]
    
    history_faqs = [
        # 历史与文化
        ("什么是文艺复兴", "文艺复兴是14-17世纪欧洲的一场思想文化运动，它标志着中世纪的结束和现代文明的开始。文艺复兴强调人文主义，重视人的价值和创造力，在文学、艺术、科学等领域产生了深远影响。"),
        ("什么是工业革命", "工业革命是18-19世纪从英国开始的重大历史变革，它标志着人类社会从农业文明向工业文明的转变。工业革命通过机械化、工厂化生产，极大地提高了生产力，改变了人类的生活方式和社会结构。"),
        ("什么是古代文明", "古代文明是人类早期发展的重要阶段，包括古埃及、古希腊、古罗马、古中国、古印度等文明。这些文明在政治、经济、文化、科技等方面取得了辉煌成就，为现代文明奠定了基础。"),
    ]
    
    economics_faqs = [
        # 经济学
        ("什么是市场经济", "市场经济是一种以市场机制配置资源的经济体制，它通过价格信号引导资源的流动和配置。市场经济强调自由竞争、私有产权、价格机制，是现代社会主要的经济组织形式。"),
        ("什么是通货膨胀", "通货膨胀是指货币供应量增加导致物价普遍上涨的经济现象。适度的通货膨胀有利于经济发展，但过度的通货膨胀会降低货币购买力，影响经济稳定和人民生活水平。"),
    ]
    
    # 定义知识缓存配置
    cache_configs = [
        {
            'name': 'ai_ml',
            'description': '人工智能与机器学习',
            'faq_pairs': ai_ml_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache_ai_ml.pth'
        },
        {
            'name': 'computer_science',
            'description': '计算机科学',
            'faq_pairs': cs_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache_cs.pth'
        },
        {
            'name': 'mathematics',
            'description': '数学',
            'faq_pairs': math_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache_math.pth'
        },
        {
            'name': 'physics',
            'description': '物理学',
            'faq_pairs': physics_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache_physics.pth'
        },
        {
            'name': 'biology',
            'description': '生物学',
            'faq_pairs': biology_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache_biology.pth'
        },
        {
            'name': 'history',
            'description': '历史与文化',
            'faq_pairs': history_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache_history.pth'
        },
        {
            'name': 'economics',
            'description': '经济学',
            'faq_pairs': economics_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache_economics.pth'
        },
        {
            'name': 'all',
            'description': '全部领域（合并）',
            'faq_pairs': ai_ml_faqs + cs_faqs + math_faqs + physics_faqs + biology_faqs + history_faqs + economics_faqs,
            'output_path': 'output/knowledge_cache/knowledge_cache.pth'
        },
    ]
    
    print("\n可用的知识缓存配置:")
    for i, config in enumerate(cache_configs, 1):
        print(f"  {i}. {config['name']:20} - {config['description']:15} ({len(config['faq_pairs'])} 条)")
    
    print("\n" + "="*70)
    print("开始构建知识缓存...")
    print("="*70)
    
    # 构建所有知识缓存
    total_success = 0
    total_skip = 0
    total_error = 0
    
    for config in cache_configs:
        print(f"\n{'='*70}")
        print(f"构建知识缓存: {config['name']} ({config['description']})")
        print(f"{'='*70}")
        print(f"FAQ数量: {len(config['faq_pairs'])}")
        print(f"输出路径: {config['output_path']}")
        
        knowledge_cache_manager, success, skip, error = build_knowledge_cache_from_faq_pairs(
            faq_pairs=config['faq_pairs'],
            output_path=config['output_path'],
            append=False  # 每个缓存文件独立创建
        )
        
        # 保存知识缓存
        output_dir = Path(config['output_path']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        knowledge_cache_manager.save(config['output_path'])
        
        print(f"\n✓ {config['name']} 知识缓存已保存: {config['output_path']}")
        print(f"  知识项数量: {len(knowledge_cache_manager.knowledge_cache)}")
        
        total_success += success
        total_skip += skip
        total_error += error
    
    print("\n" + "="*70)
    print("所有知识缓存构建完成！")
    print("="*70)
    print(f"总计: 成功 {total_success}, 跳过 {total_skip}, 失败 {total_error}")
    print(f"共生成 {len(cache_configs)} 个知识缓存文件")
    print("\n生成的知识缓存文件:")
    for config in cache_configs:
        print(f"  - {config['output_path']} ({config['description']}, {len(config['faq_pairs'])} 条)")
    
    print("\n" + "="*70)
    print("下一步操作")
    print("="*70)
    print("现在可以训练模型，使用指定的知识缓存：")
    print("  python scripts/train.py")
    print("\n默认使用: output/knowledge_cache/knowledge_cache.pth (全部领域)")


if __name__ == "__main__":
    main()

