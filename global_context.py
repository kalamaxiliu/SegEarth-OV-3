import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import joblib
import os
import json
from utils import load_local_dinov3

class GlobalContextModulator(nn.Module):
    def __init__(self, 
                 device='cuda', 
                 prototype_path="weights/inria_prototypes.pkl",
                 dinov3_path="weights/dinov3/model.safetensors",
                 co_occurrence_path="data/co_occurrence_inria.json",
                 temperature=0.1):
        super().__init__()
        self.device = device
        self.temperature = temperature
        
        # 1. 加载 DINOv3
        self.dino_model = load_local_dinov3(dinov3_path, device=device)
        for p in self.dino_model.parameters(): p.requires_grad = False
        
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. 加载原型
        if os.path.exists(prototype_path):
            data = joblib.load(prototype_path)
            self.prototypes = torch.tensor(data['centers'], dtype=torch.float32, device=device)
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        else:
            self.prototypes = None

        # 3. 加载共现矩阵
        if os.path.exists(co_occurrence_path):
            with open(co_occurrence_path, 'r') as f:
                self.co_occurrence = json.load(f)
        else:
            self.co_occurrence = {}

    def get_global_prior(self, image_pil, class_names, return_scene_info=False):
        if self.prototypes is None or not self.co_occurrence:
            priors = {c: 1.0 for c in class_names}
            if return_scene_info:
                return priors, None, None
            return priors, None

        # 1. 提取特征
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.dino_model.forward_features(img_tensor)
            global_feat = out["x_norm_clstoken"]
            global_feat = F.normalize(global_feat, p=2, dim=1)
        
        # 2. 计算相似度 & 软投票 (Soft Voting) --- 核心修改
        # [1, N]
        sims = torch.mm(global_feat, self.prototypes.t())
        
        # 使用 Softmax 将相似度转化为权重 (Temperature=0.1 让分布稍微尖锐一点，但保留次优解)
        weights = F.softmax(sims / self.temperature, dim=1).squeeze(0)
        
        priors = {}
        for c in class_names:
            weighted_factor = 0.0
            
            # 对 5 个场景的建议进行加权求和
            for i in range(len(self.prototypes)):
                scene_id = str(i)
                # 获取该场景对类别 c 的建议 (默认为 1.0)
                factor = self.co_occurrence.get(scene_id, {}).get(c, 1.0)
                weighted_factor += factor * weights[i].item()
            
            priors[c] = weighted_factor

        if not return_scene_info:
            return priors, global_feat

        eps = 1e-8
        entropy = (-weights * torch.log(weights + eps)).sum()
        scene_info = {
            "weights": weights,
            "entropy": entropy,
            "num_scenes": len(self.prototypes)
        }
        return priors, global_feat, scene_info
