#!/usr/bin/env python3
"""
检查训练数据格式，特别是换行符的使用情况
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_utils import create_sample_dataloader
from models.base_loader import Qwen3Loader
import yaml

def check_training_data():
    """检查训练数据"""
    print("="*70)
    print("检查训练数据格式")
    print("="*70)
    
    # 加载配置和tokenizer
    config_path = "configs/qwen3_0.6b_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    loader = Qwen3Loader(config_path)
    tokenizer = loader.load_tokenizer()
    
    # 创建数据加载器
    print("\n1. 创建训练数据加载器...")
    dataloader = create_sample_dataloader(
        tokenizer=tokenizer,
        batch_size=4,
        num_samples=20,  # 只检查少量样本
        max_length=512
    )
    
    # 识别换行符token ID
    newline_token_ids = set()
    for test_str in ['\n', '\r\n', '\r']:
        try:
            token_ids = tokenizer.encode(test_str, add_special_tokens=False)
            newline_token_ids.update(token_ids)
        except:
            pass
    
    print(f"\n换行符token IDs: {newline_token_ids}")
    
    # 检查数据
    print("\n2. 检查训练数据样本...")
    total_samples = 0
    samples_with_newlines = 0
    total_newline_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        texts = batch['text']
        input_ids = batch['input_ids']
        
        for i, (text, ids) in enumerate(zip(texts, input_ids)):
            total_samples += 1
            
            # 检查文本中的换行符
            newline_count = text.count('\n') + text.count('\r')
            if newline_count > 0:
                samples_with_newlines += 1
                total_newline_count += newline_count
                print(f"\n样本 {total_samples} (批次 {batch_idx}, 样本 {i}):")
                print(f"  文本: {repr(text[:100])}...")
                print(f"  换行符数量: {newline_count}")
            
            # 检查token中的换行符
            newline_tokens = sum(1 for token_id in ids.tolist() if token_id in newline_token_ids)
            if newline_tokens > 0:
                print(f"  换行符token数量: {newline_tokens}")
                # 显示token序列中的换行符位置
                token_list = ids.tolist()
                newline_positions = [j for j, tid in enumerate(token_list) if tid in newline_token_ids]
                if newline_positions:
                    print(f"  换行符token位置: {newline_positions[:10]}")  # 只显示前10个
            
            if total_samples >= 20:  # 只检查前20个样本
                break
        
        if total_samples >= 20:
            break
    
    # 统计
    print("\n" + "="*70)
    print("统计结果")
    print("="*70)
    print(f"总样本数: {total_samples}")
    print(f"包含换行符的样本数: {samples_with_newlines}")
    print(f"换行符比例: {samples_with_newlines/total_samples*100:.1f}%")
    print(f"总换行符数量: {total_newline_count}")
    if total_samples > 0:
        print(f"平均每个样本的换行符数: {total_newline_count/total_samples:.2f}")
    
    # 建议
    print("\n" + "="*70)
    print("建议")
    print("="*70)
    if samples_with_newlines / total_samples > 0.5:
        print("⚠ 警告: 超过50%的样本包含换行符，这可能导致模型过度学习换行符")
        print("建议:")
        print("1. 清理训练数据，移除不必要的换行符")
        print("2. 在训练时对换行符应用权重惩罚（已实现）")
    else:
        print("✓ 训练数据格式正常，换行符比例较低")
    
    print("\n已实施的修复:")
    print("1. ✓ 在生成时添加了重复惩罚和换行符特殊处理")
    print("2. ✓ 在训练时对换行符token应用了权重惩罚（权重: 0.3）")

if __name__ == "__main__":
    check_training_data()

