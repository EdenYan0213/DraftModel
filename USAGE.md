# 使用指南

## 两步工作流程

本项目采用**两步工作流程**，将知识缓存构建和模型训练完全分离。

---

## 步骤1: 构建知识缓存

### 方式1: 构建单个知识缓存（默认）

#### 命令
```bash
python scripts/generate_knowledge_from_model.py
```

#### 功能
- 使用预定义的FAQ问答对（30条，涵盖7个领域）
- 对每个问答对进行prefill，提取token向量
- 存储完整的问答对文本和token向量
- 生成知识缓存文件：`output/knowledge_cache/knowledge_cache.pth`

#### 输出
- 知识缓存文件：`output/knowledge_cache/knowledge_cache.pth`
- 包含：token向量、答案起始位置、问答对文本、embedding向量

#### 注意事项
- 只需要运行一次
- 如果知识缓存已存在，会跳过已有项，只添加新的
- 可以随时添加新的FAQ问答对

### 方式2: 构建多个知识缓存（按领域分类）

#### 命令
```bash
python scripts/build_multiple_knowledge_caches.py
```

#### 功能
- 按领域分别构建知识缓存文件
- 生成8个知识缓存文件（7个领域 + 1个合并）
- 每个领域独立的知识缓存，便于按需使用

#### 输出
生成以下知识缓存文件：
- `output/knowledge_cache/knowledge_cache_ai_ml.pth` - 人工智能与机器学习
- `output/knowledge_cache/knowledge_cache_cs.pth` - 计算机科学
- `output/knowledge_cache/knowledge_cache_math.pth` - 数学
- `output/knowledge_cache/knowledge_cache_physics.pth` - 物理学
- `output/knowledge_cache/knowledge_cache_biology.pth` - 生物学
- `output/knowledge_cache/knowledge_cache_history.pth` - 历史与文化
- `output/knowledge_cache/knowledge_cache_economics.pth` - 经济学
- `output/knowledge_cache/knowledge_cache.pth` - 全部领域（合并，默认使用）

#### 注意事项
- 训练脚本默认使用 `output/knowledge_cache/knowledge_cache.pth`（全部领域）
- 如果想使用特定领域的缓存，需要修改训练脚本中的路径

---

## 步骤2: 训练模型

### 命令
```bash
python scripts/train.py
```

### 前置条件
- 知识缓存文件必须存在：`output/knowledge_cache/knowledge_cache.pth`
- 如果不存在，脚本会提示错误并退出

### 功能
- 加载基础模型（Qwen3-0.6B）
- 加载知识缓存
- **自动使用知识缓存中的完整问答对作为训练数据**
- 训练草稿模型，学习如何利用知识缓存
- 保存最佳模型和定期检查点

### 输出
- 最佳模型：`output/checkpoints/best_draft_model_epochX.pth`
- 定期检查点：`output/checkpoints/draft_model_epochX.pth`（每5个epoch）

### 训练数据来源
- **自动从知识缓存中提取**：使用 `qa_pairs` 中的完整问答对
- 训练数据 = 知识缓存中的问答对数量
- 如果知识缓存有30条问答对，训练数据就是30条（会扩展倍数）

---

## 完整工作流程示例

### 第一次使用

```bash
# 1. 构建知识缓存
python scripts/generate_knowledge_from_model.py

# 2. 训练模型
python scripts/train.py

# 3. 测试推理
python scripts/inference.py
```

### 重新训练（使用已有知识缓存）

```bash
# 直接训练，使用已有的知识缓存
python scripts/train.py
```

### 添加新知识后重新训练

```bash
# 1. 修改 scripts/generate_knowledge_from_model.py 添加新的FAQ对
# 2. 重新构建知识缓存（会跳过已有项，只添加新的）
python scripts/generate_knowledge_from_model.py

# 3. 重新训练
python scripts/train.py
```

---

## 文件说明

### 知识缓存相关
- `scripts/generate_knowledge_from_model.py` - 构建单个知识缓存（使用预定义FAQ）
- `scripts/build_multiple_knowledge_caches.py` - 构建多个知识缓存（按领域分类）
- `scripts/build_knowledge_cache.py` - 旧版构建脚本（使用简单知识库）

### 训练相关
- `scripts/train.py` - 训练脚本（依赖知识缓存）

### 测试相关
- `scripts/inference.py` - 推理测试
- `scripts/compare_similarity.py` - 相似度比较

---

## 常见问题

### Q: 为什么训练数据与知识缓存不匹配？
**A**: 现在已修复！训练数据会自动从知识缓存的 `qa_pairs` 中提取，确保完全匹配。

### Q: 可以只训练不构建知识缓存吗？
**A**: 不可以。训练脚本会检查知识缓存是否存在，如果不存在会报错退出。

### Q: 如何添加新的知识？
**A**: 修改 `scripts/generate_knowledge_from_model.py` 中的 `faq_pairs` 列表，添加新的问答对，然后重新运行脚本。

### Q: 知识缓存可以重复构建吗？
**A**: 可以。脚本会检查每个问题是否已存在，只添加新的问答对。

---

## 配置说明

主要配置在 `configs/qwen3_0.6b_config.yaml`：

```yaml
# 草稿模型配置
draft_model:
  num_sampled_layers: 3  # 使用前3层

# 知识增强配置
knowledge_enhancement:
  enabled: true
  use_vector_retrieval: true

# 训练配置
training:
  batch_size: 8
  learning_rate: 3e-5
  num_epochs: 5
```

