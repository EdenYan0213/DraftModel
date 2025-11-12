from torch.utils.data import Dataset, DataLoader
from typing import List, Any

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def collate_fn(self, batch):
        """批处理函数"""
        texts = [item['text'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        # 填充
        padded_inputs = self.tokenizer.pad(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            },
            padding=True,
            return_tensors='pt'
        )
        
        return {
            'text': texts,
            'input_ids': padded_inputs['input_ids'],
            'attention_mask': padded_inputs['attention_mask']
        }

def create_sample_dataloader(tokenizer: Any, batch_size: int = 8, 
                           num_samples: int = 1000, max_length: int = 2048) -> DataLoader:
    """创建示例数据加载器"""
    
    # 示例文本数据
    sample_texts = [
        "深度学习是机器学习的一个分支，它通过多层神经网络来学习数据的层次化表示。",
        "自然语言处理是人工智能的一个重要领域，旨在让计算机理解、解释和生成人类语言。",
        "Transformer架构通过自注意力机制实现了高效的序列建模，已成为自然语言处理的主流架构。",
        "预训练语言模型如GPT系列通过在大量文本数据上进行预训练，获得了强大的语言理解和生成能力。",
        "强化学习通过试错来学习最优策略，在游戏AI、机器人控制等领域取得了显著成果。",
        "计算机视觉使计算机能够理解和分析图像和视频内容，广泛应用于安防、医疗、自动驾驶等领域。",
        "大数据技术使得我们能够处理和分析海量数据，从中提取有价值的信息和洞察。",
        "云计算提供了按需获取的计算资源，极大地降低了企业的IT成本和运维复杂度。",
        "区块链技术通过去中心化和不可篡改的特性，为数字货币和智能合约提供了安全基础。",
        "物联网将物理世界与数字世界连接起来，实现了设备的智能化和远程控制。"
    ]
    
    # 扩展样本数据
    expanded_texts = []
    for i in range(num_samples):
        base_text = sample_texts[i % len(sample_texts)]
        # 添加一些变化
        variation = f" 样本{i}: 这是关于相关主题的扩展文本。"
        expanded_texts.append(base_text + variation)
    
    # 创建数据集
    dataset = TextDataset(expanded_texts, tokenizer, max_length)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    print(f"创建数据加载器: {len(dataloader)} 批次, 每批次 {batch_size} 样本")
    
    return dataloader


def load_texts_from_file(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    从文本文件加载数据，每行一个样本
    
    Args:
        file_path: 文件路径
        encoding: 文件编码，默认utf-8
    
    Returns:
        文本列表
    """
    texts = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                texts.append(line)
    return texts


def create_dataloader_from_file(tokenizer: Any, file_path: str, 
                                batch_size: int = 8, max_length: int = 2048,
                                encoding: str = 'utf-8') -> DataLoader:
    """
    从文件创建数据加载器
    
    Args:
        tokenizer: tokenizer
        file_path: 数据文件路径（每行一个样本）
        batch_size: 批次大小
        max_length: 最大序列长度
        encoding: 文件编码
    
    Returns:
        DataLoader
    """
    print(f"从文件加载数据: {file_path}")
    texts = load_texts_from_file(file_path, encoding)
    print(f"加载了 {len(texts)} 个样本")
    
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    print(f"创建数据加载器: {len(dataloader)} 批次, 每批次 {batch_size} 样本")
    return dataloader


def create_dataloader_from_list(tokenizer: Any, texts: List[str],
                               batch_size: int = 8, max_length: int = 2048) -> DataLoader:
    """
    从文本列表创建数据加载器
    
    Args:
        tokenizer: tokenizer
        texts: 文本列表
        batch_size: 批次大小
        max_length: 最大序列长度
    
    Returns:
        DataLoader
    """
    print(f"从列表创建数据加载器: {len(texts)} 个样本")
    
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    print(f"创建数据加载器: {len(dataloader)} 批次, 每批次 {batch_size} 样本")
    return dataloader

