# 知识增强的草稿模型 (Knowledge-Enhanced Draft Model)

基于知识增强的草稿模型，用于加速大语言模型的推理。通过知识缓存和交叉注意力机制，在保持生成质量的同时提升推理速度。

## 特性

- **知识增强**: 使用知识缓存存储prefill阶段的token向量，通过相似度检索增强生成
- **交叉注意力**: 基于向量的交叉注意力机制，灵活处理不同长度的序列
- **参数高效**: 只训练前N层（默认3层），大幅减少参数量
- **速度提升**: 相比目标模型，推理速度提升约1.2-1.3倍

## 项目结构

```
CrossAndAttention/
├── models/              # 模型定义
│   ├── base_loader.py      # 基础模型加载器
│   ├── draft_model.py      # 草稿模型
│   ├── knowledge_cache.py  # 知识缓存管理器
│   └── cross_attention.py  # 交叉注意力机制
├── training/            # 训练相关
│   ├── draft_trainer.py   # 训练器
│   └── data_utils.py      # 数据工具
├── scripts/             # 脚本
│   ├── build_knowledge_cache.py  # 构建知识缓存
│   ├── train.py              # 训练脚本
│   └── inference.py         # 推理测试
├── configs/             # 配置文件
│   └── qwen3_0.6b_config.yaml
└── output/              # 输出目录
    ├── checkpoints/     # 模型检查点
    └── knowledge_cache/ # 知识缓存
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- torch
- transformers
- sentence-transformers (用于知识检索)
- tqdm
- pyyaml

### 2. 构建知识缓存（第一步）

#### 方式1: 构建单个知识缓存（默认）

```bash
python scripts/generate_knowledge_from_model.py
```

**功能**：
- 使用预定义的FAQ问答对（30条，涵盖7个领域）
- 对每个问答对进行prefill，提取token向量
- 存储完整的问答对文本和token向量
- 生成知识缓存文件：`output/knowledge_cache/knowledge_cache.pth`

#### 方式2: 构建多个知识缓存（按领域分类）

```bash
python scripts/build_multiple_knowledge_caches.py
```

**功能**：
- 按领域分别构建知识缓存文件
- 生成8个知识缓存文件（7个领域 + 1个合并）
- 每个领域独立的知识缓存，便于按需使用

**生成的文件**：
- `knowledge_cache_ai_ml.pth` - 人工智能与机器学习
- `knowledge_cache_cs.pth` - 计算机科学
- `knowledge_cache_math.pth` - 数学
- `knowledge_cache_physics.pth` - 物理学
- `knowledge_cache_biology.pth` - 生物学
- `knowledge_cache_history.pth` - 历史与文化
- `knowledge_cache_economics.pth` - 经济学
- `knowledge_cache.pth` - 全部领域（合并，默认使用）

**注意**: 
- 知识缓存只需要构建一次
- 如果已存在，会跳过已有项，只添加新的
- 可以随时添加新的FAQ问答对
- 训练脚本默认使用 `knowledge_cache.pth`（全部领域）

### 3. 训练模型（第二步）

```bash
python scripts/train.py
```

**功能**：
- 加载基础模型和知识缓存
- **自动使用知识缓存中的完整问答对作为训练数据**
- 训练草稿模型，学习如何利用知识缓存
- 保存最佳模型到 `output/checkpoints/`

**注意**: 
- 训练前必须确保知识缓存已构建完成
- 如果知识缓存不存在，脚本会报错并提示先构建

**详细使用说明**: 参见 [USAGE.md](USAGE.md)

训练配置在 `configs/qwen3_0.6b_config.yaml` 中：
- `num_sampled_layers: 3` - 选择前3层进行训练
- `batch_size: 8` - 批次大小
- `learning_rate: 3e-5` - 学习率
- `num_epochs: 5` - 训练轮数

### 4. 推理测试

```bash
python scripts/inference.py
```

## 配置说明

主要配置项：

```yaml
# 草稿模型配置
draft_model:
  num_sampled_layers: 3  # 选择前N层

# 知识增强配置
knowledge_enhancement:
  enabled: true
  use_vector_retrieval: true  # 使用向量相似度检索

# 训练配置
training:
  batch_size: 8
  learning_rate: 3e-5
  num_epochs: 5
  kl_divergence_weight: 0.5      # KL散度损失权重
  acceptance_loss_weight: 0.3    # 接受损失权重
```

## 工作原理

### 1. 知识缓存

- 存储"问题+答案"输入给基础模型后，prefill阶段输出的token向量
- 使用sentence-transformers进行相似度检索
- 每个知识项包含token向量序列和答案起始位置

### 2. 交叉注意力

- 输入：问题序列向量 + 检索到的相似序列向量
- 从向量投影得到QKV矩阵
- 支持mask答案部分（只关注答案，忽略问题部分）

### 3. 训练损失

- **KL散度损失**: 知识蒸馏，让草稿模型学习目标模型的分布
- **交叉熵损失**: 标准语言建模损失
- **接受概率损失**: 最大化目标模型对草稿模型预测的接受概率

## 性能

- **参数压缩**: 使用前3层，参数量约为目标模型的10-15%
- **速度提升**: 推理速度提升约1.2-1.3倍
- **生成质量**: 在知识相关任务上表现良好

## 注意事项

1. **数据量**: 建议至少800+训练样本，以获得稳定的训练效果
2. **知识库**: 当前知识库较小（8个知识项），可根据需要扩展
3. **设备**: 支持CPU和CUDA，MPS可能有兼容性问题

## 许可证

本项目遵循原项目的许可证。

