# 快速开始指南

## 📐 模型架构

### 草稿模型结构

草稿模型基于 Qwen3-0.6B，通过以下方式构建：

1. **层采样**: 从基础模型的28层中均匀采样6层 `[0, 5, 11, 16, 22, 27]`
2. **知识增强**: 在每个采样层后添加交叉注意力机制，利用预缓存的KV矩阵
3. **过渡层**: 在采样层之间添加过渡层，弥补语义断层

### 架构流程

```
输入文本
  ↓
Token Embedding
  ↓
┌─────────────────────────────────────┐
│ 采样层 0: Self-Attention + MLP      │
│  ↓                                  │
│  Cross-Attention (知识增强)         │
│  ↓                                  │
│  过渡层                              │
│  ↓                                  │
│ 采样层 5: Self-Attention + MLP      │
│  ↓                                  │
│  Cross-Attention (知识增强)         │
│  ↓                                  │
│  过渡层                              │
│  ... (重复到采样层 27)              │
└─────────────────────────────────────┘
  ↓
Layer Norm
  ↓
LM Head
  ↓
输出 Logits
```

### 知识增强机制

- **知识缓存**: 预计算常见问题的答案部分KV矩阵
- **交叉注意力**: 使用当前hidden states作为query，知识KV作为key/value
- **门控融合**: 动态融合原始特征和知识增强特征

---

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 构建知识缓存（首次运行）

```bash
python scripts/build_knowledge_cache.py
```

这将：
- 加载 Qwen3-0.6B 基础模型
- 对15个常见问题生成答案并提取KV矩阵
- 保存到 `output/knowledge_cache/knowledge_cache.pth`

### 3. 训练模型

```bash
python scripts/train_with_knowledge.py
```

训练配置（在 `configs/qwen3_0.6b_config.yaml` 中）：
- **训练轮数**: 5 epochs
- **批次大小**: 8
- **学习率**: 3e-5
- **知识蒸馏**: 启用（KL散度权重 0.8）
- **接受概率损失**: 启用（权重 0.3）

训练完成后，最佳模型保存在：
- `output/checkpoints/best_draft_model_knowledge_epoch5.pth`

### 4. 推理测试

```bash
python scripts/test_knowledge_questions.py
```

测试知识库中的问题，查看接受率和生成质量。

### 5. 推测解码基准测试

```bash
python scripts/benchmark_speculative_decoding.py
```

对比直接生成和推测解码的性能。

---

## 💻 代码示例

### 加载模型并推理

```python
import torch
import yaml
from models.base_loader import Qwen3Loader
from models.knowledge_enhanced_draft import Qwen3DraftModel
from models.knowledge_cache import KnowledgeCacheManager

# 1. 加载配置
with open("configs/qwen3_0.6b_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# 2. 加载基础模型
loader = Qwen3Loader("configs/qwen3_0.6b_config.yaml")
target_model = loader.load_target_model(device='cpu')
tokenizer = loader.load_tokenizer()

# 3. 加载知识缓存
cache_path = "output/knowledge_cache/knowledge_cache.pth"
cache_data = torch.load(cache_path, map_location='cpu')
knowledge_cache_manager = KnowledgeCacheManager(
    hidden_size=config['base_model']['hidden_size'],
    num_heads=config['base_model']['num_attention_heads'],
    cache_dim=config['knowledge_enhancement']['cache_dim']
)
knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})

# 4. 创建草稿模型
draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager)
draft_model.eval()

# 5. 加载训练好的权重
checkpoint = torch.load(
    "output/checkpoints/best_draft_model_knowledge_epoch5.pth",
    map_location='cpu'
)
draft_model.load_state_dict(checkpoint['model_state_dict'])

# 6. 推理
text = "深度学习是"
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    outputs = draft_model(inputs['input_ids'], retrieve_knowledge=True, query_text=text)
    logits = outputs['logits']
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print(f"输入: {text}")
    print(f"预测: {tokenizer.decode([next_token.item()])}")
```

### 使用推测解码

```python
from inference.speculative_decoder import SpeculativeDecoder

# 创建推测解码器（gamma=5表示草稿模型一次生成5个token）
decoder = SpeculativeDecoder(draft_model, target_model, tokenizer, gamma=5)

# 生成文本
input_ids = tokenizer("深度学习是", return_tensors='pt')['input_ids']
generated = decoder.generate(
    input_ids=input_ids,
    max_new_tokens=20,
    temperature=1.0,
    top_p=0.9
)

print(tokenizer.decode(generated[0]))
```

---

## 📊 模型性能

- **参数量**: 686M（比基础模型多15%，因为添加了知识增强层）
- **推理速度**: 比基础模型快约1.6倍（因为只有6层）
- **接受率**: 32-80%（取决于问题类型）

---

## ⚙️ 配置说明

主要配置文件：`configs/qwen3_0.6b_config.yaml`

### 关键配置项

```yaml
draft_model:
  num_sampled_layers: 6  # 采样层数
  sampling_strategy: "uniform"  # 采样策略

knowledge_enhancement:
  enabled: true  # 启用知识增强
  cache_dim: 512  # 缓存维度

training:
  num_epochs: 5
  batch_size: 8
  learning_rate: 3e-5
  kl_divergence_weight: 0.8  # 知识蒸馏权重
  acceptance_loss_weight: 0.3  # 接受概率损失权重
```

---

## 📝 注意事项

1. **首次运行**: 需要下载 Qwen3-0.6B 模型（约1.2GB）
2. **训练时间**: CPU训练约需1-2小时，GPU/MPS更快
3. **内存需求**: 训练时约需8-10GB内存
4. **知识缓存**: 包含15个常见问题的答案部分KV矩阵

---

## 🔗 相关文件

- **配置文件**: `configs/qwen3_0.6b_config.yaml`
- **训练脚本**: `scripts/train_with_knowledge.py`
- **测试脚本**: `scripts/test_knowledge_questions.py`
- **知识缓存构建**: `scripts/build_knowledge_cache.py`
- **推测解码器**: `inference/speculative_decoder.py`

