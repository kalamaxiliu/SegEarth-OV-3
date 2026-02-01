import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import joblib
import shutil
import sys

# ==========================================
# 引入父目录路径
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from utils import load_local_dinov3
except ImportError:
    print("[Error] Could not import 'utils.py'. Make sure it is in the project root.")
    sys.exit(1)

# ================= 配置区域 (针对 Inria 修改) =================
# 1. 数据集路径 (根据你的日志: data/Inria/train/images)
POSSIBLE_PATHS = [
    "data/Potsdam/Images",
    "data/Potsdam/img_dir/train", 
    "data/Potsdam/Train/images"
]
DATA_ROOT = None
for p in POSSIBLE_PATHS:
    full_path = os.path.join(parent_dir, p)
    if os.path.exists(full_path):
        DATA_ROOT = full_path
        break

# 如果自动检测失败，请手动修改这里
if DATA_ROOT is None:
    # 根据你的日志，这应该是正确路径
    DATA_ROOT = "/root/user/cs_tcci_penghai/mch/SegEarth-OV-3/data/Potsdam/Images"

# 2. 原型文件 (Inria 专属)
PROTOTYPE_PATH = os.path.join(parent_dir, "weights/potsdam_prototypes.pkl")

# 3. DINO 权重
DINO_WEIGHT_PATH = os.path.join(parent_dir, "weights/dinov3/model.safetensors")

# 4. 输出目录
OUTPUT_DIR = os.path.join(parent_dir, "vis_clusters_data/potsdam")

NUM_IMAGES_TO_SCAN = 1000  
TOP_K = 5 # 每类看 5 张，防止单张偶然性
# ============================================================

class SimpleDataset(Dataset):
    def __init__(self, root, transform, limit=None):
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.files = []
        # Inria 数据集通常包含子文件夹，使用递归搜索更稳健
        for ext in valid_exts:
            self.files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        
        print(f"[Dataset] Found {len(self.files)} images in {root}")
        
        if len(self.files) == 0:
            print(f"Warning: No images found in {root}. Please check path.")

        if limit and len(self.files) > limit:
            np.random.shuffle(self.files)
            self.files = self.files[:limit]
            
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), path
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return torch.zeros(3, 518, 518), ""

def main():
    print(f"=== Inria Cluster Visualization ===")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Prototype: {PROTOTYPE_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = load_local_dinov3(DINO_WEIGHT_PATH, device=device)
    
    # 2. 加载原型
    if not os.path.exists(PROTOTYPE_PATH):
        print(f"[Error] Prototype file not found at {PROTOTYPE_PATH}")
        return

    data = joblib.load(PROTOTYPE_PATH)
    centers = torch.tensor(data['centers'], device=device) 
    centers = F.normalize(centers, p=2, dim=1)
    num_clusters = centers.shape[0]
    print(f"[Prototypes] Loaded {num_clusters} centers.")
    
    # 3. 数据准备
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = SimpleDataset(DATA_ROOT, transform, limit=NUM_IMAGES_TO_SCAN)
    if len(dataset) == 0: return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # 4. 寻找 Top-K
    cluster_candidates = {i: [] for i in range(num_clusters)}
    
    print(f"Scanning {len(dataset)} images...")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, (img, paths) in enumerate(loader):
            if paths[0] == "": continue
            
            img = img.to(device)
            path = paths[0]
            
            # 兼容不同的 DINO forward 接口
            if hasattr(model, 'forward_features'):
                out = model.forward_features(img)
                feat = out["x_norm_clstoken"]
            else:
                feat = model(img)

            feat = F.normalize(feat, p=2, dim=1)
            
            # 计算相似度
            sims = torch.mm(feat, centers.t()).squeeze(0) # [Num_Clusters]
            
            for c_idx in range(num_clusters):
                score = sims[c_idx].item()
                cluster_candidates[c_idx].append((score, path))
            
            if i % 50 == 0:
                print(f"Processed {i}/{len(dataset)}...", end='\r')

    print("\nSorting and saving top matches...")

    # 5. 保存结果
    for c_idx in range(num_clusters):
        candidates = sorted(cluster_candidates[c_idx], key=lambda x: x[0], reverse=True)
        top_matches = candidates[:TOP_K]
        
        print(f"\nCluster {c_idx}:")
        for rank, (score, filepath) in enumerate(top_matches):
            filename = os.path.basename(filepath)
            print(f"  #{rank+1}: Score {score:.4f} - {filename}")
            
            dst_name = f"cluster_{c_idx}_rank_{rank+1}_{filename}"
            shutil.copy(filepath, os.path.join(OUTPUT_DIR, dst_name))

    print("\n" + "="*50)
    print(f"DONE! Output folder: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
