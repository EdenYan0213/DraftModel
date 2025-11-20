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
                 use_vector_retrieval: bool = True, embedding_model_name: str = None,
                 target_model: Optional[nn.Module] = None, tokenizer = None):
        """
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            cache_dim: 缓存维度（用于压缩存储）
            use_vector_retrieval: 是否使用向量检索（默认True）
            embedding_model_name: embedding模型名称（已废弃，现在使用target_model的嵌入层）
            target_model: 目标模型（用于使用其embed_tokens进行检索）
            tokenizer: tokenizer（用于tokenize文本）
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.cache_dim = cache_dim
        self.use_vector_retrieval = use_vector_retrieval
        
        # 存储缓存的KV矩阵
        # key: 问题/主题标识, value: (keys, values) 元组
        self.kv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # 存储压缩后的缓存（用于快速检索）
        self.compressed_cache: Dict[str, torch.Tensor] = {}
        
        # 向量检索相关
        # 注意：embeddings应该基于KV矩阵，而不是hidden states
        # 使用KV矩阵的完整序列进行检索，不进行池化
        self.knowledge_embeddings: Dict[str, torch.Tensor] = {}  # 存储知识项的完整KV序列 (num_heads, seq_len, head_dim) 或 (seq_len, hidden_size)
        self.embedding_model = None  # 外部embedding模型（已废弃）
        self.target_model = target_model  # 目标模型（用于使用其嵌入层）
        self.tokenizer = tokenizer  # tokenizer
        
        # 优先使用目标模型的嵌入层
        if target_model is not None and tokenizer is not None:
            # 检查是否有embed_tokens
            if hasattr(target_model, 'model') and hasattr(target_model.model, 'embed_tokens'):
                self.use_vector_retrieval = use_vector_retrieval
                print("✓ 使用目标模型的嵌入层进行向量检索")
            elif hasattr(target_model, 'embed_tokens'):
                self.use_vector_retrieval = use_vector_retrieval
                print("✓ 使用目标模型的嵌入层进行向量检索")
            else:
                print("⚠ 警告: 目标模型没有embed_tokens，回退到字符串匹配检索")
                self.use_vector_retrieval = False
        elif use_vector_retrieval and SENTENCE_TRANSFORMERS_AVAILABLE:
            # 回退到外部embedding模型（不推荐，但保持兼容性）
            try:
                if embedding_model_name is None:
                    embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
                
                print(f"⚠ 警告: 未提供target_model，使用外部embedding模型: {embedding_model_name}")
                print("   建议: 使用target_model的嵌入层以保持一致性")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                print("✓ 外部Embedding模型加载完成")
            except Exception as e:
                print(f"⚠ 警告: 无法加载embedding模型 {embedding_model_name}: {e}")
                print("  将回退到字符串匹配检索")
                self.use_vector_retrieval = False
                self.embedding_model = None
        else:
            self.use_vector_retrieval = False
        
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
        # 注意：embeddings应该基于KV矩阵，而不是hidden states
        # 但为了检索，我们需要一个查询表示，这里使用KV矩阵的平均值
        if self.use_vector_retrieval:
            try:
                # 使用KV矩阵计算检索用的embedding
                # KV矩阵已经存储在self.kv_cache中，格式: (num_heads, seq_len, head_dim)
                if key in self.kv_cache:
                    cached_keys, cached_values = self.kv_cache[key]
                    
                    # 将KV矩阵转换为用于检索的表示
                    # 方法：将(num_heads, seq_len, head_dim) reshape为(seq_len, hidden_size)
                    # 然后保存完整序列，不池化
                    if cached_keys.dim() == 3:
                        # (num_heads, seq_len, head_dim) -> (seq_len, num_heads, head_dim) -> (seq_len, hidden_size)
                        seq_len = cached_keys.shape[1]
                        # 转置并reshape: (num_heads, seq_len, head_dim) -> (seq_len, hidden_size)
                        keys_reshaped = cached_keys.transpose(0, 1).contiguous()  # (seq_len, num_heads, head_dim)
                        keys_reshaped = keys_reshaped.view(seq_len, -1)  # (seq_len, hidden_size)
                        
                        # 同样处理values
                        values_reshaped = cached_values.transpose(0, 1).contiguous()
                        values_reshaped = values_reshaped.view(seq_len, -1)
                        
                        # 保存KV矩阵的完整序列（用于检索）
                        # 使用keys和values的平均作为检索表示，或者分别保存
                        # 这里使用keys作为主要检索表示（因为keys包含查询信息）
                        self.knowledge_embeddings[key] = keys_reshaped.cpu()  # (seq_len, hidden_size)
                    else:
                        # 如果已经是(seq_len, hidden_size)格式，直接保存
                        self.knowledge_embeddings[key] = cached_keys.cpu()
                elif self.target_model is not None and self.tokenizer is not None:
                    # 如果没有KV缓存，使用模型生成KV矩阵
                    input_ids = self.tokenizer(key, return_tensors="pt", padding=False, truncation=True, max_length=512)
                    device = next(self.target_model.parameters()).device
                    input_ids_tensor = input_ids['input_ids'].to(device)
                    attention_mask = input_ids.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    with torch.no_grad():
                        # 获取模型的KV矩阵（最后一层Attention的KV）
                        outputs = self.target_model(input_ids=input_ids_tensor, attention_mask=attention_mask, 
                                                   use_cache=True, output_attentions=False)
                        
                        # 从past_key_values提取最后一层的KV
                        if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                            last_layer_kv = outputs.past_key_values[-1]  # 最后一层
                            k, v = last_layer_kv
                            # k, v: (batch, num_heads, seq_len, head_dim)
                            k = k.squeeze(0)  # (num_heads, seq_len, head_dim)
                            
                            # 转换为(seq_len, hidden_size)格式
                            seq_len = k.shape[1]
                            k_reshaped = k.transpose(0, 1).contiguous().view(seq_len, -1)  # (seq_len, hidden_size)
                            
                            # 保存完整序列
                            self.knowledge_embeddings[key] = k_reshaped.cpu()
                        else:
                            print(f"警告: 无法从模型提取KV矩阵 ({key[:30]}...)")
                elif self.embedding_model is not None:
                    # 回退到外部embedding模型
                    embedding = self.embedding_model.encode(key, convert_to_numpy=True)
                    self.knowledge_embeddings[key] = embedding
                else:
                    print(f"警告: 无法计算embedding，未提供target_model或embedding_model")
            except Exception as e:
                print(f"警告: 计算embedding失败 ({key[:30]}...): {e}")
                import traceback
                traceback.print_exc()
        
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
        if self.use_vector_retrieval and self.knowledge_embeddings:
            # 检查是否有target_model或embedding_model
            if (self.target_model is not None and self.tokenizer is not None) or self.embedding_model is not None:
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
            # 计算查询的KV矩阵（与知识缓存使用相同的表示）
            # 使用最后一层Attention的KV矩阵，不池化，保留完整序列
            if self.target_model is not None and self.tokenizer is not None:
                # 使用目标模型获取KV矩阵（最后一层Attention的KV）
                input_ids = self.tokenizer(query, return_tensors="pt", padding=False, truncation=True, max_length=512)
                device = next(self.target_model.parameters()).device
                input_ids_tensor = input_ids['input_ids'].to(device)
                attention_mask = input_ids.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                with torch.no_grad():
                    # 获取模型的KV矩阵
                    outputs = self.target_model(input_ids=input_ids_tensor, attention_mask=attention_mask, 
                                               use_cache=True, output_attentions=False)
                    
                    # 从past_key_values提取最后一层的KV
                    if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                        last_layer_kv = outputs.past_key_values[-1]  # 最后一层
                        k, v = last_layer_kv
                        # k: (batch, num_heads, seq_len, head_dim)
                        k = k.squeeze(0)  # (num_heads, seq_len, head_dim)
                        
                        # 转换为(seq_len, hidden_size)格式，与知识缓存保持一致
                        seq_len = k.shape[1]
                        query_kv_sequence = k.transpose(0, 1).contiguous().view(seq_len, -1).cpu()  # (seq_len, hidden_size)
                    else:
                        raise ValueError("无法从模型提取KV矩阵")
            elif self.embedding_model is not None:
                # 回退到外部embedding模型（不推荐）
                query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
                # 转换为torch tensor以保持一致性
                query_kv_sequence = torch.from_numpy(query_embedding).unsqueeze(0)  # (1, embedding_dim)
            else:
                return None
            
            # 计算与所有知识项的相似度（使用完整序列，不池化）
            similarities = []
            
            # 使用完整KV序列计算相似度
            for key, cached_kv_seq in self.knowledge_embeddings.items():
                # cached_kv_seq: (seq_len_cached, hidden_size) - 完整序列
                # query_kv_sequence: (seq_len_query, hidden_size) - 完整序列
                
                # 计算序列级相似度（不池化）
                # 方法：计算两个序列的平均向量的相似度
                # 注意：虽然计算平均，但我们保存的是完整序列，可以根据需要改进
                query_mean = query_kv_sequence.mean(dim=0).numpy()  # (hidden_size,)
                cached_mean = cached_kv_seq.mean(dim=0).numpy()  # (hidden_size,)
                similarity = self._cosine_similarity(query_mean, cached_mean)
                
                # 未来可以改进：使用DTW（动态时间规整）或其他序列对齐方法
                # 或者使用注意力机制计算序列对齐相似度
                
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
        if self.use_vector_retrieval:
            if self.target_model is not None:
                return "vector_similarity (target_model hidden_states)"
            elif self.embedding_model is not None:
                return "vector_similarity (external embedding model)"
        return "string_match"
    
    def _recompute_all_embeddings(self):
        """重新计算所有知识项的embeddings（当维度不匹配时）"""
        if not (self.target_model is not None and self.tokenizer is not None):
            print("⚠ 警告: 无法重新计算embeddings，未提供target_model和tokenizer")
            return
        
        print(f"重新计算 {len(self.kv_cache)} 个知识项的embeddings（使用KV矩阵）...")
        keys_to_recompute = list(self.kv_cache.keys())
        device = next(self.target_model.parameters()).device
        
        for key in keys_to_recompute:
            try:
                # 使用与add_knowledge相同的方法重新计算
                input_ids = self.tokenizer(key, return_tensors="pt", padding=True, truncation=True, max_length=512)
                input_ids_tensor = input_ids['input_ids'].to(device)
                attention_mask = input_ids.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                with torch.no_grad():
                    # 使用KV矩阵重新计算（与add_knowledge保持一致）
                    # 优先使用已有的KV缓存
                    if key in self.kv_cache:
                        cached_keys, cached_values = self.kv_cache[key]
                        
                        # 将KV矩阵转换为(seq_len, hidden_size)格式
                        if cached_keys.dim() == 3:
                            seq_len = cached_keys.shape[1]
                            keys_reshaped = cached_keys.transpose(0, 1).contiguous().view(seq_len, -1)
                            self.knowledge_embeddings[key] = keys_reshaped.cpu()
                        else:
                            self.knowledge_embeddings[key] = cached_keys.cpu()
                    else:
                        # 如果没有KV缓存，从模型生成KV矩阵
                        outputs = self.target_model(input_ids=input_ids_tensor, attention_mask=attention_mask, 
                                                   use_cache=True, output_attentions=False)
                        
                        if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                            last_layer_kv = outputs.past_key_values[-1]
                            k, v = last_layer_kv
                            k = k.squeeze(0)
                            
                            seq_len = k.shape[1]
                            k_reshaped = k.transpose(0, 1).contiguous().view(seq_len, -1)
                            self.knowledge_embeddings[key] = k_reshaped.cpu()
            except Exception as e:
                print(f"警告: 重新计算embedding失败 ({key[:30]}...): {e}")
        
        print(f"✓ 重新计算完成，共 {len(self.knowledge_embeddings)} 个embeddings")

