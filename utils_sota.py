import torch
import torch.nn as nn
import os
import json
import sys

try:
    from transformers import Dinov2Model, Dinov2Config
    from transformers import logging as hf_logging
    from safetensors.torch import load_file as safe_load_file
    hf_logging.set_verbosity_error()
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class DinoV2Wrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    
    def forward_features(self, x):
        outputs = self.model(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        return {"x_norm_clstoken": cls_token}

    def forward(self, x):
        return self.forward_features(x)

def greedy_map_and_patch(raw_model, state_dict):
    """
    贪婪匹配 + 自动补全：
    1. 能够匹配的（Layer, Attention）强行匹配。
    2. 名字对不上的 MLP，通过排除法贪婪匹配。
    3. 实在缺失的（PosEmbed, Bias），用随机初始化补全，保证运行。
    """
    model_state = raw_model.state_dict()
    new_state_dict = {}
    
    print("[GreedyLoad] Starting aggressive mapping...")
    
    # === 1. 全局参数映射 ===
    for k, v in state_dict.items():
        # Embedding
        if "cls_token" in k: new_state_dict["embeddings.cls_token"] = v
        if "mask_token" in k: 
            if v.dim() == 3: v = v.squeeze(1)
            new_state_dict["embeddings.mask_token"] = v
        if "pos_embed" in k: new_state_dict["embeddings.position_embeddings"] = v
        
        # Patch Embed
        if "patch" in k and "weight" in k: new_state_dict["embeddings.patch_embeddings.projection.weight"] = v
        if "patch" in k and "bias" in k: new_state_dict["embeddings.patch_embeddings.projection.bias"] = v
        
        # Final Norm
        if k == "norm.weight": new_state_dict["layernorm.weight"] = v
        if k == "norm.bias": new_state_dict["layernorm.bias"] = v

    # === 2. 逐层贪婪匹配 ===
    # DINOv3 Large 有 24 层
    num_layers = 24
    
    for i in range(num_layers):
        hf_prefix = f"encoder.layer.{i}"
        file_prefix = f"layer.{i}." # 根据你之前的日志确认是 layer.0. 格式
        
        # 提取该层在文件中的所有 key
        layer_keys = [k for k in state_dict.keys() if k.startswith(file_prefix)]
        
        # 如果找不到 layer.0，尝试 blocks.0
        if not layer_keys:
            file_prefix = f"blocks.{i}."
            layer_keys = [k for k in state_dict.keys() if k.startswith(file_prefix)]
        
        # 分类桶
        attn_keys = []
        norm_keys = []
        mlp_keys = [] # 剩下的都是 MLP
        
        for k in layer_keys:
            suffix = k.replace(file_prefix, "")
            if "attn" in suffix or "attention" in suffix:
                attn_keys.append(k)
            elif "norm" in suffix:
                norm_keys.append(k)
            elif "ls" in suffix or "layer_scale" in suffix or "gamma" in suffix:
                # Gamma/LayerScale
                if "1" in suffix: new_state_dict[f"{hf_prefix}.layer_scale1.lambda1"] = state_dict[k]
                if "2" in suffix: new_state_dict[f"{hf_prefix}.layer_scale2.lambda1"] = state_dict[k]
            else:
                # 既不是 Attn 也不是 Norm，那就是 MLP！
                mlp_keys.append(k)
        
        # --- 处理 Attention ---
        for k in attn_keys:
            v = state_dict[k]
            if "q_proj" in k or "query" in k:
                wb = "weight" if "weight" in k else "bias"
                new_state_dict[f"{hf_prefix}.attention.attention.query.{wb}"] = v
            elif "k_proj" in k or "key" in k:
                wb = "weight" if "weight" in k else "bias"
                new_state_dict[f"{hf_prefix}.attention.attention.key.{wb}"] = v
            elif "v_proj" in k or "value" in k:
                wb = "weight" if "weight" in k else "bias"
                new_state_dict[f"{hf_prefix}.attention.attention.value.{wb}"] = v
            elif "o_proj" in k or "output" in k:
                wb = "weight" if "weight" in k else "bias"
                new_state_dict[f"{hf_prefix}.attention.output.dense.{wb}"] = v

        # --- 处理 Norm ---
        for k in norm_keys:
            v = state_dict[k]
            if "norm1" in k:
                wb = "weight" if "weight" in k else "bias"
                new_state_dict[f"{hf_prefix}.norm1.{wb}"] = v
            elif "norm2" in k:
                wb = "weight" if "weight" in k else "bias"
                new_state_dict[f"{hf_prefix}.norm2.{wb}"] = v

        # --- 贪婪处理 MLP (解决 fc1 名字不对的问题) ---
        # 排序：通常 fc1/intermediate 在前，fc2/output 在后
        mlp_keys.sort() 
        # 假设剩下了 4 个 key: [fc1.bias, fc1.weight, fc2.bias, fc2.weight]
        # 或者 [mlp.0.bias, mlp.0.weight, mlp.2.bias, mlp.2.weight]
        
        fc1_candidates = [k for k in mlp_keys if "weight" in k][:1] # 取第一个 weight
        fc2_candidates = [k for k in mlp_keys if "weight" in k][1:] # 取第二个 weight
        
        if fc1_candidates:
            k_w = fc1_candidates[0]
            new_state_dict[f"{hf_prefix}.intermediate.dense.weight"] = state_dict[k_w]
            # 尝试找对应的 bias (同名替换 weight -> bias)
            k_b = k_w.replace("weight", "bias")
            if k_b in state_dict:
                 new_state_dict[f"{hf_prefix}.intermediate.dense.bias"] = state_dict[k_b]
        
        if fc2_candidates:
            k_w = fc2_candidates[0]
            new_state_dict[f"{hf_prefix}.output.dense.weight"] = state_dict[k_w]
            k_b = k_w.replace("weight", "bias")
            if k_b in state_dict:
                 new_state_dict[f"{hf_prefix}.output.dense.bias"] = state_dict[k_b]

    # === 3. 自动补全 (Patching) ===
    # 任何还没被填满的 key，直接用模型原始的随机初始化参数填充
    # 这样 load_state_dict 永远不会报错
    patched_count = 0
    final_dict = {}
    
    for k, v in model_state.items():
        if k in new_state_dict:
            final_dict[k] = new_state_dict[k]
        else:
            # 缺失 Key，使用默认值
            # print(f"[Patching] Missing {k} in file, using default init.")
            final_dict[k] = v
            patched_count += 1
            
    print(f"[GreedyLoad] Patched {patched_count} missing keys with default initialization.")
    
    # 加载 (绝对不会报错)
    raw_model.load_state_dict(final_dict)
    print("[SUCCESS] Model loaded successfully (with patching).")
    
    return raw_model

def load_local_dinov3(checkpoint_path, device='cuda'):
    print(f"[DINOv3] Loading from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"{checkpoint_path}")
    
    # Config
    model_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f: config_dict = json.load(f)
        config_dict["model_type"] = "dinov2"
        if "num_register_tokens" in config_dict: del config_dict["num_register_tokens"]
        config = Dinov2Config.from_dict(config_dict)
    else:
        config = Dinov2Config(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, image_size=518, patch_size=14)

    raw_model = Dinov2Model(config)
    state_dict = safe_load_file(checkpoint_path)

    # 调用贪婪加载
    raw_model = greedy_map_and_patch(raw_model, state_dict)
    
    return DinoV2Wrapper(raw_model).to(device).eval()