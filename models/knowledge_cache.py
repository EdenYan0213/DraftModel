"""
知识缓存管理器 - 缓存常见问题的KV矩阵
支持向量相似度检索
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装，将使用字符串匹配检索")

class KnowledgeCacheManager:
    """知识缓存管理器 - 存储和检索常见问题的KV矩阵"""
    
    def __init__(self, hidden_size: int, num_heads: int, cache_dim: int = 512, 
                 use_vector_retrieval: bool = True, embedding_model_name: str = None):
        """
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            cache_dim: 缓存维度（用于压缩存储）
            use_vector_retrieval: 是否使用向量检索（默认True）
            embedding_model_name: embedding模型名称（默认使用中文模型）
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.cache_dim = cache_dim
        self.use_vector_retrieval = use_vector_retrieval and SENTENCE_TRANSFORMERS_AVAILABLE
        
        # 存储缓存的KV矩阵
        # key: 问题/主题标识, value: (keys, values) 元组
        self.kv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # 存储压缩后的缓存（用于快速检索）
        self.compressed_cache: Dict[str, torch.Tensor] = {}
        
        # 向量检索相关
        self.knowledge_embeddings: Dict[str, np.ndarray] = {}  # 存储知识项的向量
        self.embedding_model = None
        
        if self.use_vector_retrieval:
            try:
                # 使用中文embedding模型（如果没有指定，使用轻量级模型）
                if embedding_model_name is None:
                    # 使用轻量级中文模型
                    embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
                    # 如果上面的模型不可用，尝试其他模型
                    # embedding_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                
                print(f"加载embedding模型: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                print("✓ Embedding模型加载完成")
            except Exception as e:
                print(f"⚠ 警告: 无法加载embedding模型 {embedding_model_name}: {e}")
                print("  将回退到字符串匹配检索")
                self.use_vector_retrieval = False
                self.embedding_model = None
        
        # KV投影层（用于压缩和解压缩）
        self.kv_compressor = nn.Sequential(
            nn.Linear(hidden_size * 2, cache_dim),
            nn.GELU(),
            nn.Linear(cache_dim, cache_dim)
        )
        
        self.kv_decompressor = nn.Sequential(
            nn.Linear(cache_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2)
        )
    
    def add_knowledge(self, 
                      key: str, 
                      keys: torch.Tensor, 
                      values: torch.Tensor,
                      compress: bool = True):
        """
        添加知识到缓存
        
        Args:
            key: 知识标识（如问题文本或主题）
            keys: Key矩阵 (seq_len, hidden_size) 或 (num_heads, seq_len, head_dim)
            values: Value矩阵 (seq_len, hidden_size) 或 (num_heads, seq_len, head_dim)
            compress: 是否压缩存储
        """
        # 确保keys和values的形状一致
        if keys.dim() == 2:
            # (seq_len, hidden_size) -> 需要转换为多头格式
            keys = keys.unsqueeze(0).expand(self.num_heads, -1, -1)
            values = values.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        # 存储原始KV
        self.kv_cache[key] = (keys.clone(), values.clone())
        
        # 计算并存储embedding（用于向量检索）
        if self.use_vector_retrieval and self.embedding_model is not None:
            try:
                embedding = self.embedding_model.encode(key, convert_to_numpy=True)
                self.knowledge_embeddings[key] = embedding
            except Exception as e:
                print(f"警告: 计算embedding失败 ({key[:30]}...): {e}")
        
        # 压缩存储（可选）
        if compress:
            try:
                # 将KV合并并压缩
                # keys: (num_heads, seq_len, head_dim) -> mean over heads -> (seq_len, head_dim)
                # values: (num_heads, seq_len, head_dim) -> mean over heads -> (seq_len, head_dim)
                keys_mean = keys.mean(dim=0)  # (seq_len, head_dim)
                values_mean = values.mean(dim=0)  # (seq_len, head_dim)
                
                # 展平并合并
                keys_flat = keys_mean.flatten()  # (seq_len * head_dim,)
                values_flat = values_mean.flatten()  # (seq_len * head_dim,)
                kv_combined = torch.cat([keys_flat, values_flat], dim=0)  # (seq_len * head_dim * 2,)
                
                # 如果长度不匹配，使用平均池化
                expected_size = self.hidden_size * 2
                if kv_combined.size(0) != expected_size:
                    # 使用线性插值或平均池化调整大小
                    if kv_combined.size(0) > expected_size:
                        # 平均池化
                        kv_combined = kv_combined.view(1, -1, expected_size).mean(dim=1).squeeze(0)
                    else:
                        # 填充或重复
                        repeat_factor = (expected_size + kv_combined.size(0) - 1) // kv_combined.size(0)
                        kv_combined = kv_combined.repeat(repeat_factor)[:expected_size]
                
                # 确保形状正确: (hidden_size*2,)
                kv_combined = kv_combined.view(expected_size)
                
                kv_compressed = self.kv_compressor(kv_combined)  # (cache_dim,)
                self.compressed_cache[key] = kv_compressed
            except Exception as e:
                # 如果压缩失败，只存储原始KV，不压缩
                print(f"警告: 压缩知识缓存失败 ({key[:30]}...): {e}")
                pass

    def retrieve(self, query: str, topk: int = 1, threshold: float = 0.5) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        检索知识KV矩阵（优先使用向量检索，失败时回退到字符串匹配）
        
        Args:
            query: 查询文本
            topk: 返回top-k个最相关的知识（当前实现只返回top-1）
            threshold: 相似度阈值（0-1），低于此阈值不返回
        
        Returns:
            (keys, values) 元组，如果未找到返回None
        """
        # 优先使用向量检索
        if self.use_vector_retrieval and self.embedding_model is not None and self.knowledge_embeddings:
            result = self.retrieve_by_similarity(query, topk=topk, threshold=threshold)
            if result is not None:
                return result
        
        # 回退到字符串匹配
        return self.retrieve_by_string_match(query)
    
    def retrieve_by_similarity(self, query: str, topk: int = 1, threshold: float = 0.5) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        基于向量相似度检索知识
        
        Args:
            query: 查询文本
            topk: 返回top-k个最相关的知识
            threshold: 相似度阈值
        
        Returns:
            (keys, values) 元组，如果未找到返回None
        """
        if not self.knowledge_embeddings:
            return None
        
        try:
            # 计算查询的embedding
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # 计算与所有知识项的相似度
            similarities = []
            for key, emb in self.knowledge_embeddings.items():
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, emb)
                if similarity >= threshold:
                    similarities.append((key, similarity))
            
            if not similarities:
                return None
            
            # 按相似度排序，返回Top-K
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 返回最相似的知识项
            best_key = similarities[0][0]
            keys, values = self.kv_cache[best_key]
            
            if len(similarities) > 1:
                print(f"检索: '{query[:30]}...' -> '{best_key[:30]}...' (相似度: {similarities[0][1]:.3f})")
            
            return keys, values
            
        except Exception as e:
            print(f"警告: 向量检索失败: {e}，回退到字符串匹配")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def retrieve_by_string_match(self, query: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        基于字符串匹配检索知识（回退方法）
        
        Args:
            query: 查询文本
        
        Returns:
            (keys, values) 元组，如果未找到返回None
        """
        query_lower = query.lower()
        
        # 查找匹配的缓存
        matches = []
        for cache_key in self.kv_cache.keys():
            if query_lower in cache_key.lower() or cache_key.lower() in query_lower:
                matches.append(cache_key)
        
        if not matches:
            # 如果没有精确匹配，返回第一个
            if self.kv_cache:
                first_key = list(self.kv_cache.keys())[0]
                keys, values = self.kv_cache[first_key]
                return keys, values
            return None
        
        # 返回第一个匹配的KV
        best_match = matches[0]
        keys, values = self.kv_cache[best_match]
        return keys, values
    
    def retrieve_compressed(self, query: str) -> Optional[torch.Tensor]:
        """
        检索压缩的知识缓存
        
        Args:
            query: 查询文本
        
        Returns:
            压缩的缓存向量 (cache_dim,)，如果未找到返回None
        """
        query_lower = query.lower()
        
        for cache_key in self.compressed_cache.keys():
            if query_lower in cache_key.lower() or cache_key.lower() in query_lower:
                return self.compressed_cache[cache_key]
        
        # 如果没有匹配，返回第一个
        if self.compressed_cache:
            return list(self.compressed_cache.values())[0]
        
        return None
    
    def build_cache_from_model(self, 
                              model: nn.Module,
                              common_questions: List[str],
                              tokenizer,
                              device: str = "cuda"):
        """
        从目标模型构建知识缓存
        
        Args:
            model: 目标模型（用于生成KV）
            common_questions: 常见问题列表
            tokenizer: tokenizer
            device: 设备
        """
        model.eval()
        
        print(f"开始构建知识缓存，共 {len(common_questions)} 个问题...")
        
        with torch.no_grad():
            for i, question in enumerate(common_questions):
                # Tokenize问题
                inputs = tokenizer(question, return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                
                # 获取所有层的KV
                # 这里简化处理，只获取最后一层的KV
                # 实际可以获取所有层的KV并聚合
                outputs = model(input_ids=input_ids, output_attentions=False, use_cache=True)
                
                # 从模型的past_key_values中提取KV
                # 注意：不同模型结构可能不同，这里需要根据实际模型调整
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                    # 获取最后一层的KV
                    past_kv = outputs.past_key_values[-1]  # 最后一层
                    keys = past_kv[0]  # (batch, num_heads, seq_len, head_dim)
                    values = past_kv[1]
                    
                    # 压缩维度：(batch, num_heads, seq_len, head_dim) -> (num_heads, seq_len, head_dim)
                    keys = keys.squeeze(0)
                    values = values.squeeze(0)
                    
                    # 添加到缓存
                    self.add_knowledge(question, keys, values, compress=True)
                
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{len(common_questions)} 个问题")
        
        print(f"知识缓存构建完成，共缓存 {len(self.kv_cache)} 个知识项")
    
    def clear(self):
        """清空缓存"""
        self.kv_cache.clear()
        self.compressed_cache.clear()
        self.knowledge_embeddings.clear()
    
    def __len__(self):
        """返回缓存的知识数量"""
        return len(self.kv_cache)
    
    def get_retrieval_method(self) -> str:
        """返回当前使用的检索方法"""
        if self.use_vector_retrieval and self.embedding_model is not None:
            return "vector_similarity"
        return "string_match"

