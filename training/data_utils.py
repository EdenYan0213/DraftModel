"""
数据工具
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class KnowledgeDataset(Dataset):
    """知识增强数据集"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'text': text
        }


def create_sample_dataloader(tokenizer, batch_size: int = 8, max_length: int = 2048, 
                            knowledge_cache_manager=None):
    """创建示例数据加载器
    
    Args:
        tokenizer: tokenizer
        batch_size: batch size
        max_length: max sequence length
        knowledge_cache_manager: 知识缓存管理器，如果提供，将使用知识缓存中的问答对作为训练数据
    """
    
    # 如果提供了知识缓存，使用知识缓存中的问答对
    if knowledge_cache_manager is not None and len(knowledge_cache_manager.knowledge_cache) > 0:
        print(f"使用知识缓存中的问答对作为训练数据")
        sample_texts = []
        
        # 从知识缓存中提取完整的问答对
        if hasattr(knowledge_cache_manager, 'qa_pairs') and len(knowledge_cache_manager.qa_pairs) > 0:
            # 使用存储的完整问答对
            for question, (q, a) in knowledge_cache_manager.qa_pairs.items():
                # 组合成完整的训练文本：问题 + 空格 + 答案（与构建知识缓存时的格式一致）
                # 注意：这里需要与 extract_prefill_vectors 中的格式保持一致
                # 如果构建时是 question + answer（无空格），这里也应该是 q + a
                # 但为了更好的tokenization，建议在问题和答案之间加空格
                full_text = q + " " + a if not q.endswith(" ") and not a.startswith(" ") else q + a
                sample_texts.append(full_text)
            print(f"  从qa_pairs中提取了 {len(sample_texts)} 条完整问答对")
            
            # 检查是否有知识项没有对应的qa_pairs
            cache_keys = set(knowledge_cache_manager.knowledge_cache.keys())
            qa_keys = set(knowledge_cache_manager.qa_pairs.keys())
            missing_keys = cache_keys - qa_keys
            if missing_keys:
                print(f"  ⚠ 警告: 有 {len(missing_keys)} 个知识项没有对应的qa_pairs，这些项将不会用于训练")
                print(f"     建议重新构建知识缓存以确保所有知识项都有对应的qa_pairs")
        else:
            # 如果没有qa_pairs，尝试从问题构建（向后兼容）
            print(f"  警告: 知识缓存中没有qa_pairs，使用问题作为训练数据")
            for question in knowledge_cache_manager.knowledge_cache.keys():
                sample_texts.append(question)
        
        print(f"  共 {len(sample_texts)} 条训练样本")
    else:
        # 使用默认的训练数据
        sample_texts = [
            "深度学习是人工智能的一个分支，它使用多层神经网络来学习数据的特征表示。",
            "自然语言处理是人工智能的一个分支，它主要研究如何让计算机理解并处理人类语言。",
            "计算机视觉是计算机科学与工程的交叉学科，它涉及到图像处理、模式识别、机器学习和人工智能等技术。",
            "强化学习是人工智能的一个重要分支，它在很多领域都有广泛的应用，包括自动驾驶、机器人、医疗诊断等。",
            "Transformer架构是一种基于注意力机制的神经网络架构，它在自然语言处理领域取得了突破性进展。",
            "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。",
            "注意力机制是神经网络中用于解决长距离依赖问题的一种机制，它允许模型在处理序列时关注不同位置的信息。",
            "神经网络是一种模拟人脑神经元结构的计算模型，它由多个层级的节点（神经元）组成，通过权重连接。",
        ]
    
    # 扩展数据集（根据数据量调整扩展倍数）
    if len(sample_texts) < 50:
        # 如果数据量少，多扩展一些
        expansion_factor = max(50, 100 // len(sample_texts))
    else:
        # 如果数据量多，少扩展一些
        expansion_factor = max(10, 50 // len(sample_texts))
    
    expanded_texts = sample_texts * expansion_factor
    print(f"训练数据: {len(sample_texts)} 条样本，扩展 {expansion_factor} 倍 = {len(expanded_texts)} 条")
    
    dataset = KnowledgeDataset(expanded_texts, tokenizer, max_length)
    
    # 80%训练集，20%验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

