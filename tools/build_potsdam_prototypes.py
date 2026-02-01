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

# === 配置区域 ===
# 请根据你的实际路径修改
DATA_ROOT = "data/Potsdam/Images"  # 或者 data/Potsdam/Train/images
OUTPUT_FILE = "weights/potsdam_prototypes.pkl"
DINO_WEIGHT_PATH = "weights/dinov3/model.safetensors"
N_CLUSTERS = 4  # 建议 4 类：1.密集建筑区 2.住宅+庭院 3.纯植被 4.杂乱/其他
# ================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils import load_local_dinov3
except ImportError:
    def load_local_dinov3(path, device):
        import timm
        return timm.create_model('vit_large_patch14_dinov2', pretrained=False)

class SimpleDataset(Dataset):
    def __init__(self, root, transform, max_images=2000):
        self.files = glob.glob(os.path.join(root, "*.tif")) + \
                     glob.glob(os.path.join(root, "*.png")) + \
                     glob.glob(os.path.join(root, "*.jpg"))
        # 递归查找
        if len(self.files) == 0:
            self.files = glob.glob(os.path.join(root, "**/*.tif"), recursive=True)

        print(f"Found {len(self.files)} images.")
        if len(self.files) > max_images:
            np.random.shuffle(self.files)
            self.files = self.files[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img), self.files[idx]
        except:
            return torch.zeros(3, 518, 518), ""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DINOv3 from {DINO_WEIGHT_PATH}...")
    model = load_local_dinov3(DINO_WEIGHT_PATH, device=device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SimpleDataset(DATA_ROOT, transform)
    if len(dataset) == 0:
        print("Error: No images found. Check DATA_ROOT.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    print("Extracting features...")
    features_list = []
    sample_paths = []

    with torch.no_grad():
        for i, (imgs, paths) in enumerate(loader):
            imgs = imgs.to(device)
            if hasattr(model, 'forward_features'):
                out = model.forward_features(imgs)
                cls_token = out["x_norm_clstoken"]
            else:
                cls_token = model(imgs)
            features_list.append(cls_token.cpu().numpy())
            sample_paths.extend(paths)

    all_features = np.concatenate(features_list, axis=0)
    
    print(f"Clustering {all_features.shape[0]} samples into {N_CLUSTERS} centers...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_features)
    
    prototypes = {
        "centers": kmeans.cluster_centers_,
        "labels": list(range(N_CLUSTERS)),
        "repr_images": {}
    }

    # 找出代表性图片
    print("\n=== Representative Images for Potsdam Clusters ===")
    for i in range(N_CLUSTERS):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            cluster_feats = all_features[cluster_indices]
            center = kmeans.cluster_centers_[i]
            dists = np.linalg.norm(cluster_feats - center, axis=1)
            min_dist_idx = np.argmin(dists)
            real_idx = cluster_indices[min_dist_idx]
            path = sample_paths[real_idx]
            prototypes["repr_images"][str(i)] = path
            print(f"Cluster {i}: {path}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    joblib.dump(prototypes, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
