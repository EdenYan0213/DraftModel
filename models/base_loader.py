import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import yaml
from typing import Dict, Any

class Qwen3Loader:
    """Qwen3-0.6B 模型加载器"""
    
    def __init__(self, config_path: str = "configs/qwen3_0.6b_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['base_model']['name']
        self.target_model = None
        self.tokenizer = None
        
    def load_target_model(self, device: str = "auto") -> nn.Module:
        """加载目标模型"""
        print(f"加载目标模型: {self.model_name}")
        
        try:
            # 设备选择：优先CUDA，其次MPS，最后CPU
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            print(f"使用设备: {device}")
            
            # MPS不支持float16，需要使用float32
            dtype = torch.float32 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32)
            
            # 对于MPS，不使用device_map，手动移动
            if device == "mps":
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map=None,  # MPS不支持device_map
                    trust_remote_code=True
                )
                # 手动移动到MPS
                self.target_model = self.target_model.to(device)
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map=device if device == "cuda" else None,
                    trust_remote_code=True
                )
                # 确保在正确的设备上
                if device == "cpu":
                    self.target_model = self.target_model.cpu()
                elif device == "cuda" and torch.cuda.is_available():
                    if not hasattr(self.target_model, 'device') or 'cuda' not in str(self.target_model.device):
                        self.target_model = self.target_model.cuda()
                    
        except Exception as e:
            print(f"使用自动设备映射失败: {e}")
            print("回退到CPU设备...")
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.target_model = self.target_model.cpu()
        
        self.target_model.eval()
        print(f"目标模型加载完成，层数: {len(self.target_model.model.layers)}")
        
        return self.target_model
    
    def load_tokenizer(self) -> Any:
        """加载tokenizer"""
        print(f"加载tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Tokenizer加载完成，词汇表大小: {len(self.tokenizer)}")
        return self.tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        if self.target_model is None:
            self.load_target_model()
        
        info = {
            'model_name': self.model_name,
            'total_layers': len(self.target_model.model.layers),
            'hidden_size': self.target_model.config.hidden_size,
            'num_attention_heads': self.target_model.config.num_attention_heads,
            'num_key_value_heads': self.target_model.config.num_key_value_heads,
            'vocab_size': self.target_model.config.vocab_size,
            'intermediate_size': self.target_model.config.intermediate_size
        }
        
        return info
    
    def validate_config(self) -> bool:
        """验证配置与模型匹配"""
        model_info = self.get_model_info()
        base_config = self.config['base_model']
        
        mismatches = []
        for key, expected in base_config.items():
            if key in model_info and model_info[key] != expected:
                mismatches.append(f"{key}: 配置值={expected}, 实际值={model_info[key]}")
        
        if mismatches:
            print("配置不匹配警告:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            return False
        
        print("配置验证通过")
        return True

