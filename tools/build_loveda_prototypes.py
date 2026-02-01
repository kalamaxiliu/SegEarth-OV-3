import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
import joblib
import sys

# 将父目录加入 path 以便导入 utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_local_dinov3

# ================= 配置区域 =================
# 修改为你的图片文件夹路径 (LoveDA 或 Inria 的 train/images)
DATA_ROOT = "data/LoveDA/Train/images" 
OUTPUT_FILE = "weights/scene_prototypes.pkl"
DINO_WEIGHT_PATH = "weights/dinov3_vitl14.pth" # 你的本地权重路径
N_CLUSTERS = 5 
# ===========================================

class SimpleDataset(Dataset):
    def __init__(self, root, transform):
        # 支持常见格式
        self.files = glob.glob(os.path.join(root, "*.png")) + \
                     glob.glob(os.path.join(root, "*.tif")) + \
                     glob.glob(os.path.join(root, "*.jpg"))
        
        # 为了速度，最多只随机采样 2000 张图做聚类即可，不需要全部
        if len(self.files) > 2000:
            np.random.shuffle(self.files)
            self.files = self.files[:2000]
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root}")
            
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(3, 518, 518) # Dummy return

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = load_local_dinov3(DINO_WEIGHT_PATH, device=device)

    # 2. 数据预处理 (DINOv2/v3 推荐 518x518)
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SimpleDataset(DATA_ROOT, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Extracting features from {len(dataset)} images...")
    features_list = []
    
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if i % 10 == 0: print(f"Processing batch {i}...")
            imgs = imgs.to(device)
            # DINOv2/v3 forward 返回的是 dict 或者 Tensor，视具体版本
            # ViT-L 通常直接通过 forward_features 取 x_norm_clstoken 比较稳
            out = model.forward_features(imgs)
            cls_token = out["x_norm_clstoken"] # [B, 1024]
            features_list.append(cls_token.cpu().numpy())

    all_features = np.concatenate(features_list, axis=0)
    
    # 3. 聚类
    print(f"Clustering features shape: {all_features.shape} into {N_CLUSTERS} centers...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    kmeans.fit(all_features)
    
    prototypes = {
        "centers": kmeans.cluster_centers_,  # [N, 1024]
        "labels": list(range(N_CLUSTERS))
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    joblib.dump(prototypes, OUTPUT_FILE)
    print(f"Success! Prototypes saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()