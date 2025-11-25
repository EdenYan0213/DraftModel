"""
知识缓存管理器 - 存储prefill阶段的token向量
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装，将使用字符串匹配检索")


class KnowledgeCacheManager:
    """
    知识缓存管理器
    
    存储"问题+答案"输入给基础模型后，基础模型在prefill阶段输出的token向量
    每个token的向量维度为 hidden_size
    """
    
    def __init__(self, hidden_size: int, 
                 use_vector_retrieval: bool = True,
                 embedding_model_name: str = None,
                 target_model: Optional[nn.Module] = None,
                 tokenizer = None):
        self.hidden_size = hidden_size
        self.use_vector_retrieval = use_vector_retrieval
        
        # 存储知识项
        self.knowledge_cache: Dict[str, torch.Tensor] = {}  # key: 问题, value: (seq_len, hidden_size) token向量序列
        self.answer_start_indices: Dict[str, int] = {}  # 答案在序列中的起始位置
        self.knowledge_embeddings: Dict[str, np.ndarray] = {}  # 用于相似度检索的embedding
        self.qa_pairs: Dict[str, Tuple[str, str]] = {}  # key: 问题, value: (问题, 答案) 完整问答对
        
        # Embedding模型（用于相似度检索）
        self.embedding_model = None
        if use_vector_retrieval:
            if embedding_model_name is None:
                embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.embedding_model = SentenceTransformer(embedding_model_name)
                    print(f"✓ Embedding模型加载完成: {embedding_model_name}")
                except Exception as e:
                    print(f"⚠ 无法加载embedding模型: {e}")
                    self.use_vector_retrieval = False
            else:
                print("⚠ sentence-transformers未安装，禁用向量检索")
                self.use_vector_retrieval = False
        
        self.target_model = target_model
        self.tokenizer = tokenizer
    
    def add_knowledge(self, 
                     key: str,
                     question: str,
                     answer: str,
                     token_vectors: torch.Tensor,
                     answer_start_idx: int):
        """添加知识项"""
        self.knowledge_cache[key] = token_vectors.clone()
        self.answer_start_indices[key] = answer_start_idx
        self.qa_pairs[key] = (question, answer)  # 存储完整问答对
        
        # 计算并存储embedding（用于相似度检索）
        if self.use_vector_retrieval and self.embedding_model is not None:
            full_text = question + answer
            embedding = self.embedding_model.encode(full_text, convert_to_numpy=True)
            self.knowledge_embeddings[key] = embedding
            print(f"✓ 已添加知识项: {key} (序列长度: {token_vectors.shape[0]}, 答案起始位置: {answer_start_idx})")
        else:
            print(f"✓ 已添加知识项: {key} (序列长度: {token_vectors.shape[0]}, 答案起始位置: {answer_start_idx})")
    
    def retrieve_by_similarity(self, 
                              query: str,
                              top_k: int = 1,
                              threshold: float = 0.5,
                              device: Optional[torch.device] = None) -> Optional[Tuple[torch.Tensor, int]]:
        """
        基于相似度检索知识
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果
            threshold: 相似度阈值
            device: 目标设备（可选），如果提供，返回的张量将移动到该设备
        
        Returns:
            (token_vectors, answer_start_idx) 或 None
        """
        if not self.use_vector_retrieval or self.embedding_model is None:
            return None
        
        if not self.knowledge_embeddings:
            return None
        
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            similarities = []
            for key, cached_embedding in self.knowledge_embeddings.items():
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity >= threshold:
                    similarities.append((key, similarity))
            
            if not similarities:
                return None
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_key, best_similarity = similarities[0]
            
            token_vectors = self.knowledge_cache[best_key]
            answer_start_idx = self.answer_start_indices[best_key]
            
            # 如果指定了设备，将张量移动到该设备
            if device is not None:
                token_vectors = token_vectors.to(device)
            
            return token_vectors, answer_start_idx
            
        except Exception as e:
            print(f"⚠ 相似度检索失败: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def save(self, filepath: str):
        """保存知识缓存"""
        data = {
            'knowledge_cache': self.knowledge_cache,
            'answer_start_indices': self.answer_start_indices,
            'knowledge_embeddings': self.knowledge_embeddings,
            'qa_pairs': self.qa_pairs,  # 保存完整问答对
            'hidden_size': self.hidden_size,
            'use_vector_retrieval': self.use_vector_retrieval
        }
        torch.save(data, filepath)
        print(f"✓ 知识缓存已保存: {filepath}")
    
    def load(self, filepath: str, device: str = 'auto'):
        """
        加载知识缓存
        
        Args:
            filepath: 缓存文件路径
            device: 加载设备，'auto'自动选择，'cuda'使用CUDA，'cpu'使用CPU
        """
        from .utils import get_device
        load_device = get_device(device)
        # 加载时使用CPU，避免设备不匹配，后续使用时再移动到目标设备
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        self.knowledge_cache = data.get('knowledge_cache', {})
        self.answer_start_indices = data.get('answer_start_indices', {})
        self.knowledge_embeddings = data.get('knowledge_embeddings', {})
        self.qa_pairs = data.get('qa_pairs', {})  # 加载完整问答对
        self.hidden_size = data.get('hidden_size', self.hidden_size)
        self.use_vector_retrieval = data.get('use_vector_retrieval', self.use_vector_retrieval)
        print(f"✓ 知识缓存已加载: {filepath}")
        print(f"  共 {len(self.knowledge_cache)} 个知识项")

