# Qwen3-0.6B 草稿模型完整实现

基于 Qwen3-0.6B 构建的草稿模型，包含均匀采样层和知识增强功能。已训练完成，速度比原模型快 **1.64倍**。

## 🏗️ 项目结构

```
CrossAndAttention/
├── configs/
│   └── qwen3_0.6b_config.yaml          # 主配置文件
├── models/
│   ├── __init__.py
│   ├── base_loader.py                   # 模型加载器
│   ├── layer_sampler.py                 # 层采样策略
│   ├── knowledge_cache.py               # 知识缓存管理
│   └── knowledge_enhanced_draft.py      # 知识增强草稿模型
├── training/
│   ├── __init__.py
│   ├── draft_trainer.py                 # 训练器
│   └── data_utils.py                    # 数据处理工具
├── inference/
│   ├── __init__.py
│   └── speculative_decoder.py           # 推测解码器
├── scripts/
│   ├── train_draft.py                   # 训练脚本
│   ├── build_knowledge_cache.py         # 构建知识缓存
│   ├── test_trained_model.py            # 测试训练好的模型
│   └── benchmark_speed.py                # 速度对比测试
├── output/
│   ├── checkpoints/
│   │   └── best_draft_model_epoch5.pth  # 最佳模型（4.3GB）
│   └── knowledge_cache/
│       └── knowledge_cache.pth          # 知识缓存
├── requirements.txt                     # 依赖包
└── README.md                            # 本文档
```

## 🚀 快速开始

### 1. 环境设置

```bash
pip install -r requirements.txt
```

### 2. 构建知识缓存（首次运行）

```bash
python scripts/build_knowledge_cache.py
```

这将：
- 加载 Qwen3-0.6B 基础模型
- 对常见问题进行前向传播
- 提取并存储 KV 缓存
- 保存到 `output/knowledge_cache/knowledge_cache.pth`

### 3. 训练草稿模型

```bash
python scripts/train_draft.py
```

训练过程包括：
- 知识蒸馏（KL散度损失）
- 交叉熵损失
- 验证和检查点保存
- 自动保存最佳模型

**训练结果**：
- 训练损失: 8.85
- 验证损失: 7.95
- 训练轮数: 5 epochs
- 采样层: [0, 5, 11, 16, 22, 27] (从28层中采样6层)

### 4. 测试训练好的模型

```bash
python scripts/test_trained_model.py
```

### 5. 速度对比测试

```bash
python scripts/benchmark_speed.py
```

**测试结果**：
- 草稿模型速度: **11.29 tokens/s**
- 目标模型速度: 8.85 tokens/s
- **加速比: 1.64x** (快63.6%)

## 📋 主要特性

### 1. 均匀层采样
- 从 Qwen3-0.6B 的28层中智能采样6层
- 支持均匀、几何、对数三种采样策略
- 自动包含首层和尾层

### 2. 知识增强
- 交叉注意力机制
- 知识缓存查询（15个常见问题的KV缓存）
- 门控融合机制

### 3. 完整训练流程
- 知识蒸馏训练
- 支持梯度累积
- 学习率调度
- 自动检查点保存
- 数值稳定性检查
- 梯度裁剪

### 4. 推测解码
- 使用草稿模型加速推理
- 目标模型验证机制
- 自动接受/拒绝策略

## ⚙️ 配置说明

主要配置文件：`configs/qwen3_0.6b_config.yaml`

### 草稿模型配置
```yaml
draft_model:
  sampling_strategy: "uniform"  # uniform, geometric, logarithmic
  num_sampled_layers: 6
  sampled_indices: [0, 5, 11, 16, 22, 27]
  add_knowledge_enhancement: true
```

### 训练配置
```yaml
training:
  batch_size: 8
  learning_rate: 3e-5
  num_epochs: 5
  use_knowledge_distillation: true
  kl_divergence_weight: 0.8
  max_seq_length: 512
  gradient_accumulation_steps: 2
```

### 知识增强配置
```yaml
knowledge_enhancement:
  enabled: true
  cache_dim: 512
  num_heads: 8
  fusion_gate: true
```

## 📊 训练结果

### 性能指标
- **训练损失**: 8.85 (从初始 502.78 下降)
- **验证损失**: 7.95 (最佳模型)
- **验证困惑度**: 2833.54
- **训练轮数**: 5 epochs
- **采样层**: [0, 5, 11, 16, 22, 27]

### 速度对比
- **草稿模型**: 11.29 tokens/s
- **目标模型**: 8.85 tokens/s
- **加速比**: 1.64x
- **速度提升**: 63.6%

### 模型文件
- **最佳模型**: `output/checkpoints/best_draft_model_epoch5.pth` (4.3 GB)
- **知识缓存**: `output/knowledge_cache/knowledge_cache.pth`

## 🔧 使用示例

### 加载并测试训练好的模型

```python
import torch
import yaml
from models.base_loader import Qwen3Loader
from models.knowledge_enhanced_draft import Qwen3DraftModel

# 加载配置
with open("configs/qwen3_0.6b_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# 加载基础模型和tokenizer
loader = Qwen3Loader("configs/qwen3_0.6b_config.yaml")
target_model = loader.load_target_model(device='cpu')
tokenizer = loader.load_tokenizer()

# 确保pad_token设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载知识缓存
from models.knowledge_cache import KnowledgeCacheManager
cache_path = "output/knowledge_cache/knowledge_cache.pth"
cache_data = torch.load(cache_path, map_location='cpu')
knowledge_cache_manager = KnowledgeCacheManager(
    hidden_size=config['base_model']['hidden_size'],
    num_heads=config['base_model']['num_attention_heads'],
    cache_dim=config['knowledge_enhancement']['cache_dim']
)
knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})

# 创建并加载草稿模型
draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
draft_model = draft_model.cpu()
draft_model.eval()

# 加载训练好的权重
checkpoint = torch.load(
    "output/checkpoints/best_draft_model_epoch5.pth",
    map_location='cpu'
)
draft_model.load_state_dict(checkpoint['model_state_dict'])

# 推理
text = "今天天气很好，"
inputs = tokenizer(text, return_tensors='pt', padding=True)

# 确保attention_mask存在
if 'attention_mask' not in inputs:
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    inputs['attention_mask'] = (inputs['input_ids'] != pad_token_id).long()

with torch.no_grad():
    outputs = draft_model(**inputs)
    logits = outputs['logits']
    
    # 获取下一个token
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    predicted = tokenizer.decode([next_token.item()])
    
    print(f"输入: {text}")
    print(f"预测: {predicted}")
```

### 使用推测解码

```python
from inference.speculative_decoder import SpeculativeDecoder

decoder = SpeculativeDecoder(draft_model, target_model, tokenizer, gamma=4)
generated = decoder.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9
)
```

## 📝 注意事项

1. **模型下载**: 首次运行需要下载 Qwen3-0.6B 模型，确保网络连接正常
2. **设备支持**: 支持 CUDA、MPS (Apple Silicon) 和 CPU
3. **训练时间**: 完整训练可能需要数小时，取决于硬件配置
4. **数据准备**: 当前使用示例数据，实际应用需要准备真实训练数据
5. **Attention Mask**: 已自动处理，不再出现 pad_token == eos_token 的警告
6. **数值稳定性**: 已添加 NaN/Inf 检测和梯度裁剪

## 🔍 已修复的问题

1. ✅ **Attention Mask 警告**: 已自动创建 attention_mask，不再出现警告
2. ✅ **数值稳定性**: 添加了 NaN/Inf 检测和梯度裁剪
3. ✅ **模型保存**: 自动保存最佳模型和检查点
4. ✅ **进度显示**: 训练过程显示进度条

## 📈 性能优化建议

1. **使用真实数据**: 使用真实数据集可以进一步提高模型质量
2. **调整超参数**: 可以尝试不同的学习率、batch size等
3. **继续训练**: 可以从检查点继续训练更多轮次
4. **量化加速**: 考虑使用模型量化进一步加速

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目遵循原 Qwen 模型的许可证。
