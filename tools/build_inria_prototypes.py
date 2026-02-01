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

# === 路径配置 ===
# 请确保这里指向 Inria 数据集的图片目录
# 建议使用 Train 目录，或者 Train+Val 的一部分
DATA_ROOT = "data/Inria/train/images" 
OUTPUT_FILE = "weights/inria_prototypes.pkl"
DINO_WEIGHT_PATH = "weights/dinov3/model.safetensors" 
N_CLUSTERS = 4  # 建议设为 4 类：1.密集城市 2.稀疏住宅 3.植被/荒野 4.由于Inria切片后可能有纯黑背景或其他
# ================

# 引入项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils import load_local_dinov3
except ImportError:
    # 简单的 fallback，防止找不到 utils
    def load_local_dinov3(path, device):
        print("Loading DINOv3 from hub (fallback)...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        return model.to(device)

class InriaSamplerDataset(Dataset):
    def __init__(self, root, transform, max_images=3000):
        # Inria 图像通常是 .tif
        self.files = glob.glob(os.path.join(root, "*.tif")) + \
                     glob.glob(os.path.join(root, "*.png"))
        
        if len(self.files) == 0:
            # 尝试递归搜索
            self.files = glob.glob(os.path.join(root, "**/*.tif"), recursive=True)

        print(f"Found {len(self.files)} images.")
        
        # 随机采样，避免处理所有图片太慢
        if len(self.files) > max_images:
            np.random.seed(42)
            np.random.shuffle(self.files)
            self.files = self.files[:max_images]
            
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img), self.files[idx]
        except Exception as e:
            return torch.zeros(3, 518, 518), ""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    print(f"Loading DINOv3 from {DINO_WEIGHT_PATH}...")
    model = load_local_dinov3(DINO_WEIGHT_PATH, device=device)
    model.eval()

    # 2. 预处理
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = InriaSamplerDataset(DATA_ROOT, transform)
    if len(dataset) == 0:
        print("Error: No images found. Check DATA_ROOT.")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Extracting features...")
    features_list = []
    
    # 用于后续可视化的采样路径（可选）
    sample_paths = []

    with torch.no_grad():
        for i, (imgs, paths) in enumerate(loader):
            if i % 5 == 0: print(f"Batch {i}/{len(loader)}")
            imgs = imgs.to(device)
            # 兼容不同的 DINOv3 调用方式
            if hasattr(model, 'forward_features'):
                out = model.forward_features(imgs)
                cls_token = out["x_norm_clstoken"]
            else:
                cls_token = model(imgs)
                
            features_list.append(cls_token.cpu().numpy())
            sample_paths.extend(paths)

    all_features = np.concatenate(features_list, axis=0)
    
    # 3. K-Means 聚类
    print(f"Clustering {all_features.shape[0]} samples into {N_CLUSTERS} centers...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=666, n_init=10)
    labels = kmeans.fit_predict(all_features)
    
    # 4. 保存原型
    prototypes = {
        "centers": kmeans.cluster_centers_,  # [N, Dim]
        "labels": list(range(N_CLUSTERS)),
        # 保存每个聚类中心距离最近的一张图的路径，方便你人工确认该类是什么场景！
        "repr_images": {} 
    }

    # 找到每个聚类中心的代表性图片
    print("\n=== Cluster Analysis (Please verify manually) ===")
    for i in range(N_CLUSTERS):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            # 计算该类内所有点到中心的距离
            cluster_feats = all_features[cluster_indices]
            center = kmeans.cluster_centers_[i]
            dists = np.linalg.norm(cluster_feats - center, axis=1)
            min_dist_idx = np.argmin(dists)
            repr_idx = cluster_indices[min_dist_idx]
            
            prototypes["repr_images"][str(i)] = sample_paths[repr_idx]
            print(f"Cluster {i}: Representative Image -> {sample_paths[repr_idx]}")
        else:
            print(f"Cluster {i}: Empty")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    joblib.dump(prototypes, OUTPUT_FILE)
    print(f"\nSaved Inria prototypes to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()