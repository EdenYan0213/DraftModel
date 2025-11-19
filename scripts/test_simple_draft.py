#!/usr/bin/env python3
"""
测试简单草稿模型 - 只使用前3层，层间插入交叉注意力
"""

import os
import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.simple_draft_model import SimpleDraftModel


def main():
    print("="*70)
    print("简单草稿模型测试")
    print("="*70)
    
    config_path = "configs/qwen3_0.6b_config.yaml"
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载基础模型
    print("\n1. 加载基础模型...")
    loader = Qwen3Loader(config_path)
    target_model = loader.load_target_model(device='cpu')
    tokenizer = loader.load_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 基础模型加载完成，总层数: {len(target_model.model.layers)}")
    
    # 创建简单草稿模型
    print("\n2. 创建简单草稿模型（前3层 + 层间交叉注意力）...")
    simple_draft = SimpleDraftModel(config, target_model)
    simple_draft = simple_draft.cpu()
    simple_draft.eval()
    target_model.eval()
    
    print("✓ 简单草稿模型创建完成")
    
    # 测试推理
    print("\n3. 测试推理...")
    test_prompts = [
        "深度学习是",
        "自然语言处理是",
        "机器学习是"
    ]
    
    print("\n" + "="*70)
    print("生成结果对比")
    print("="*70)
    
    for prompt in test_prompts:
        print(f"\n问题: {prompt}")
        print("-"*70)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids']
        
        # 简单草稿模型生成
        print("简单草稿模型生成中...")
        with torch.no_grad():
            draft_outputs = simple_draft(input_ids)
            draft_logits = draft_outputs['logits'][:, -1, :]
            draft_token_id = torch.argmax(draft_logits, dim=-1).item()
            draft_token = tokenizer.decode([draft_token_id])
            print(f"  预测token: {draft_token!r} (ID: {draft_token_id})")
        
        # 基础模型生成
        print("基础模型生成中...")
        with torch.no_grad():
            target_outputs = target_model(input_ids)
            target_logits = target_outputs.logits[:, -1, :]
            target_token_id = torch.argmax(target_logits, dim=-1).item()
            target_token = tokenizer.decode([target_token_id])
            print(f"  预测token: {target_token!r} (ID: {target_token_id})")
        
        # 比较
        match = "✓" if draft_token_id == target_token_id else "✗"
        print(f"  匹配: {match}")
    
    # 测试完整生成
    print("\n" + "="*70)
    print("完整句子生成测试")
    print("="*70)
    
    test_prompt = "深度学习是"
    print(f"\n问题: {test_prompt}")
    
    inputs = tokenizer(test_prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    # 简单草稿模型生成完整句子
    print("\n简单草稿模型生成（最多20个token）...")
    with torch.no_grad():
        generated_ids = simple_draft.generate(input_ids, max_new_tokens=20)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(test_prompt):
            generated_text = generated_text[len(test_prompt):].strip()
        print(f"生成结果: {generated_text}")
    
    # 基础模型生成完整句子
    print("\n基础模型生成（最多20个token）...")
    with torch.no_grad():
        generated_ids = target_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(test_prompt):
            generated_text = generated_text[len(test_prompt):].strip()
        print(f"生成结果: {generated_text}")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
    print("\n说明:")
    print("- 简单草稿模型使用基础模型的前3层")
    print("- 在每层之间插入了交叉注意力机制")
    print("- 模型不需要训练，直接使用基础模型的权重")
    print("- 交叉注意力层是随机初始化的（未训练）")


if __name__ == "__main__":
    main()

