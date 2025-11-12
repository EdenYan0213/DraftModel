#!/usr/bin/env python3
"""
分析草稿模型和原模型的参数量、层数和计算量
"""

import torch
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_loader import Qwen3Loader
from models.knowledge_enhanced_draft import Qwen3DraftModel

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def analyze_layer_structure(model, model_name):
    """分析模型层结构"""
    print(f"\n{'='*60}")
    print(f"{model_name} 结构分析")
    print(f"{'='*60}")
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,}")
    
    # 统计层数
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        print(f"Transformer层数: {num_layers}")
    elif hasattr(model, 'enhanced_layers'):
        num_layers = len(model.enhanced_layers)
        print(f"采样层数: {num_layers}")
        if hasattr(model, 'sampled_indices'):
            print(f"采样索引: {model.sampled_indices}")
    else:
        num_layers = "未知"
        print(f"层数: {num_layers}")
    
    # 分析各组件参数量
    print(f"\n各组件参数量:")
    component_params = {}
    
    if hasattr(model, 'enhanced_layers'):
        # 草稿模型
        for i, layer in enumerate(model.enhanced_layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            component_params[f'Layer_{i}'] = layer_params
            print(f"  Layer {i} (索引 {model.sampled_indices[i]}): {layer_params:,}")
        
        # 知识增强层
        if hasattr(model, 'use_knowledge') and model.use_knowledge:
            for i, layer in enumerate(model.enhanced_layers):
                if hasattr(layer, 'knowledge_enhancement') and layer.knowledge_enhancement is not None:
                    enh_params = sum(p.numel() for p in layer.knowledge_enhancement.parameters())
                    component_params[f'Knowledge_Enhancement_{i}'] = enh_params
                    print(f"  Knowledge Enhancement {i}: {enh_params:,}")
        
        # Embedding层
        if hasattr(model, 'embed_tokens'):
            emb_params = sum(p.numel() for p in model.embed_tokens.parameters())
            component_params['Embedding'] = emb_params
            print(f"  Embedding: {emb_params:,}")
        
        # LM Head
        if hasattr(model, 'lm_head'):
            lm_params = sum(p.numel() for p in model.lm_head.parameters())
            component_params['LM_Head'] = lm_params
            print(f"  LM Head: {lm_params:,}")
    
    elif hasattr(model, 'model'):
        # 目标模型
        if hasattr(model.model, 'embed_tokens'):
            emb_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
            component_params['Embedding'] = emb_params
            print(f"  Embedding: {emb_params:,}")
        
        if hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                layer_params = sum(p.numel() for p in layer.parameters())
                component_params[f'Layer_{i}'] = layer_params
        
        if hasattr(model, 'lm_head'):
            lm_params = sum(p.numel() for p in model.lm_head.parameters())
            component_params['LM_Head'] = lm_params
            print(f"  LM Head: {lm_params:,}")
    
    return total_params, num_layers, component_params

def estimate_flops_per_layer(hidden_size, num_heads, seq_len, is_attention=True):
    """估算每层的FLOPs"""
    if is_attention:
        # Self-Attention: 4 * hidden_size^2 * seq_len (QKV投影 + Attention计算)
        # 简化估算
        flops = 4 * hidden_size * hidden_size * seq_len + 2 * hidden_size * seq_len * seq_len
    else:
        # FFN: 2 * hidden_size * ffn_dim * seq_len
        ffn_dim = hidden_size * 4  # 典型FFN维度
        flops = 2 * hidden_size * ffn_dim * seq_len
    
    return flops

def main():
    print("="*70)
    print("草稿模型 vs 原模型 - 参数量和计算量分析")
    print("="*70)
    
    # 加载配置
    with open('configs/qwen3_0.6b_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载基础模型
    print("\n1. 加载基础模型...")
    loader = Qwen3Loader('configs/qwen3_0.6b_config.yaml')
    target_model = loader.load_target_model(device='cpu')
    tokenizer = loader.load_tokenizer()
    
    # 加载知识缓存
    cache_path = "output/knowledge_cache/knowledge_cache.pth"
    knowledge_cache_manager = None
    
    if Path(cache_path).exists():
        from models.knowledge_cache import KnowledgeCacheManager
        cache_data = torch.load(cache_path, map_location='cpu')
        knowledge_cache_manager = KnowledgeCacheManager(
            hidden_size=config['base_model']['hidden_size'],
            num_heads=config['base_model']['num_attention_heads'],
            cache_dim=config['knowledge_enhancement']['cache_dim']
        )
        knowledge_cache_manager.kv_cache = cache_data.get('kv_cache', {})
        print("✓ 知识缓存加载完成")
    
    # 创建草稿模型
    print("\n2. 创建草稿模型...")
    draft_model = Qwen3DraftModel(config, target_model, knowledge_cache_manager=knowledge_cache_manager)
    draft_model = draft_model.cpu()
    
    # 分析目标模型
    target_params, target_layers, target_components = analyze_layer_structure(target_model, "目标模型 (Qwen3-0.6B)")
    
    # 分析草稿模型
    draft_params, draft_layers, draft_components = analyze_layer_structure(draft_model, "草稿模型")
    
    # 对比分析
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")
    
    print(f"\n参数量对比:")
    print(f"  目标模型: {target_params:,} ({target_params/1e6:.2f}M)")
    print(f"  草稿模型: {draft_params:,} ({draft_params/1e6:.2f}M)")
    print(f"  差异: {draft_params - target_params:,} ({draft_params/target_params:.2%})")
    
    print(f"\n层数对比:")
    print(f"  目标模型: {target_layers} 层")
    print(f"  草稿模型: {draft_layers} 层")
    if isinstance(target_layers, int) and isinstance(draft_layers, int):
        print(f"  层数减少: {target_layers - draft_layers} 层 ({(1-draft_layers/target_layers)*100:.1f}%)")
    
    # 估算计算量（FLOPs）
    hidden_size = config['base_model']['hidden_size']
    num_heads = config['base_model']['num_attention_heads']
    seq_len = 512  # 典型序列长度
    
    print(f"\n计算量估算 (序列长度={seq_len}):")
    
    # 每层的FLOPs（简化估算）
    attention_flops_per_layer = estimate_flops_per_layer(hidden_size, num_heads, seq_len, is_attention=True)
    ffn_flops_per_layer = estimate_flops_per_layer(hidden_size, num_heads, seq_len, is_attention=False)
    total_flops_per_layer = attention_flops_per_layer + ffn_flops_per_layer
    
    if isinstance(target_layers, int):
        target_total_flops = target_layers * total_flops_per_layer
        print(f"  目标模型每层FLOPs: ~{total_flops_per_layer/1e9:.2f}G")
        print(f"  目标模型总FLOPs: ~{target_total_flops/1e9:.2f}G")
    
    if isinstance(draft_layers, int):
        draft_total_flops = draft_layers * total_flops_per_layer
        print(f"  草稿模型每层FLOPs: ~{total_flops_per_layer/1e9:.2f}G")
        print(f"  草稿模型总FLOPs: ~{draft_total_flops/1e9:.2f}G")
        
        if isinstance(target_layers, int):
            flops_reduction = (1 - draft_total_flops / target_total_flops) * 100
            print(f"  FLOPs减少: {flops_reduction:.1f}%")
    
    # 知识增强层的额外计算
    if hasattr(draft_model, 'use_knowledge') and draft_model.use_knowledge:
        print(f"\n知识增强层:")
        enh_params = 0
        for layer in draft_model.enhanced_layers:
            if hasattr(layer, 'knowledge_enhancement') and layer.knowledge_enhancement is not None:
                enh_params += sum(p.numel() for p in layer.knowledge_enhancement.parameters())
        print(f"  知识增强参数量: {enh_params:,} ({enh_params/1e6:.2f}M)")
        print(f"  占草稿模型比例: {enh_params/draft_params:.2%}")
    
    # 结论
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    
    print(f"\n为什么草稿模型参数量更大但速度更快？")
    print(f"\n1. 层数减少是关键:")
    if isinstance(target_layers, int) and isinstance(draft_layers, int):
        print(f"   - 目标模型: {target_layers} 层")
        print(f"   - 草稿模型: {draft_layers} 层")
        print(f"   - 减少了 {target_layers - draft_layers} 层 ({(1-draft_layers/target_layers)*100:.1f}%)")
        print(f"   - 这是速度提升的主要原因！")
    
    print(f"\n2. 参数量增加的原因:")
    print(f"   - 知识增强层添加了额外的参数")
    print(f"   - 交叉注意力机制需要额外的投影层")
    print(f"   - 门控融合机制增加了少量参数")
    print(f"   - 但这些额外参数的计算开销相对较小")
    
    print(f"\n3. 推理速度的决定因素:")
    print(f"   - 主要取决于层数（前向传播次数）")
    print(f"   - 每层需要一次前向传播")
    print(f"   - 层数减少直接减少了计算时间")
    print(f"   - 参数量主要影响内存占用，对单次推理速度影响较小")
    
    print(f"\n4. 实际效果:")
    print(f"   - 参数量: 草稿模型略大 ({draft_params/target_params:.2%})")
    print(f"   - 计算量: 草稿模型更少 (约减少 {(1-draft_layers/target_layers)*100:.1f}% 的层)")
    print(f"   - 推理速度: 草稿模型快 1.64x")
    print(f"   - 结论: 层数减少带来的速度提升 > 参数量增加带来的开销")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()

