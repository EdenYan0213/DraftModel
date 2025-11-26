"""
工具函数模块
"""

import torch
from typing import Union


def get_device(device: Union[str, None] = 'auto') -> str:
    """
    自动选择设备
    
    Args:
        device: 设备选择，'auto'自动选择，'cuda'使用CUDA，'cpu'使用CPU，None则自动选择
    
    Returns:
        str: 设备名称 ('cuda', 'cpu', 或 'mps')
    """
    if device is None:
        device = 'auto'
    
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    # 如果指定了设备，验证是否可用
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ 警告: CUDA不可用，回退到CPU")
        return 'cpu'
    elif device == 'mps' and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
        print("⚠ 警告: MPS不可用，回退到CPU")
        return 'cpu'
    
    return device


def print_device_info(device: str):
    """打印设备信息"""
    if device == 'cuda':
        print(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    elif device == 'mps':
        print(f"使用设备: MPS (Apple Silicon)")
    else:
        print(f"使用设备: CPU")

