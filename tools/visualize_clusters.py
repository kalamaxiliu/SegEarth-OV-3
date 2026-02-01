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
# 关键：引入父目录路径，以便加载我们修好的 utils.py
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from utils import load_local_dinov3
except ImportError:
    print("[Error] Could not import 'utils.py'. Make sure it is in the project root.")
    sys.exit(1)

# ================= 配置区域 =================
# 自动检测 LoveDA 路径 (兼容 img_dir 和 images_png)
POSSIBLE_PATHS = [
    "data/LoveDA/Train/img_dir",
    "data/LoveDA/Train/images_png",
    "data/LoveDA/Train/images"
]
DATA_ROOT = None
for p in POSSIBLE_PATHS:
    if os.path.exists(os.path.join(parent_dir, p)):
        DATA_ROOT = os.path.join(parent_dir, p)
        break

if DATA_ROOT is None:
    # 如果找不到，请手动在这里填入你的绝对路径
    DATA_ROOT = "/root/user/cs_tcci_penghai/mch/SegEarth-OV-3/data/LoveDA/Train/img_dir"

PROTOTYPE_PATH = os.path.join(parent_dir, "weights/scene_prototypes.pkl")
DINO_WEIGHT_PATH = os.path.join(parent_dir, "weights/dinov3/model.safetensors")
OUTPUT_DIR = os.path.join(parent_dir, "vis_clusters")
NUM_IMAGES_TO_SCAN = 1000  # 扫描多少张图片来寻找代表 (越多越准，但越慢)
TOP_K = 3 # 每个聚类保存几张代表图
# ===========================================

class SimpleDataset(Dataset):
    def __init__(self, root, transform, limit=None):
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.files = []
        for ext in valid_exts:
            self.files.extend(glob.glob(os.path.join(root, f"*{ext}")))
        
        print(f"[Dataset] Found {len(self.files)} images in {root}")
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root}. Check DATA_ROOT config.")

        if limit and len(self.files) > limit:
            # 随机打乱后取前 limit 张，保证样本多样性
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
    print(f"Using Data Root: {DATA_ROOT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型 (复用 utils 的强力加载逻辑)
    model = load_local_dinov3(DINO_WEIGHT_PATH, device=device)
    
    # 2. 加载原型
    if not os.path.exists(PROTOTYPE_PATH):
        print(f"[Error] Prototype file not found at {PROTOTYPE_PATH}")
        print("Please run 'tools/build_scene_prototypes.py' first.")
        return

    data = joblib.load(PROTOTYPE_PATH)
    centers = torch.tensor(data['centers'], device=device) # [N_Clusters, Dim]
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # 4. 寻找 Top-K
    # 结构: {cluster_idx: [(score, filepath), ...]}
    cluster_candidates = {i: [] for i in range(num_clusters)}
    
    print(f"Scanning {len(dataset)} images to find best cluster representatives...")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, (img, paths) in enumerate(loader):
            if paths[0] == "": continue
            
            img = img.to(device)
            path = paths[0]
            
            # 提取特征
            out = model.forward_features(img)
            feat = out["x_norm_clstoken"] # [1, 1024]
            feat = F.normalize(feat, p=2, dim=1)
            
            # 计算相似度
            sims = torch.mm(feat, centers.t()).squeeze(0) # [Num_Clusters]
            
            # 记录数据
            for c_idx in range(num_clusters):
                score = sims[c_idx].item()
                cluster_candidates[c_idx].append((score, path))
            
            if i % 50 == 0:
                print(f"Processed {i}/{len(dataset)} images...", end='\r')

    print("\nSorting and saving top matches...")

    # 5. 排序并保存
    for c_idx in range(num_clusters):
        # 按分数降序排列
        candidates = sorted(cluster_candidates[c_idx], key=lambda x: x[0], reverse=True)
        top_matches = candidates[:TOP_K]
        
        print(f"\nCluster {c_idx}:")
        for rank, (score, filepath) in enumerate(top_matches):
            filename = os.path.basename(filepath)
            print(f"  #{rank+1}: Score {score:.4f} - {filename}")
            
            # 复制并重命名图片
            # 格式: vis_clusters/cluster_0_rank_1_score_0.95.png
            dst_name = f"cluster_{c_idx}_rank_{rank+1}_score_{score:.3f}.png"
            shutil.copy(filepath, os.path.join(OUTPUT_DIR, dst_name))

    print("\n" + "="*50)
    print(f"DONE! Please open the folder: {OUTPUT_DIR}")
    print("="*50)
    print("Next Step: Open these images, determine what scene each cluster represents,")
    print("and edit 'data/co_occurrence.json' accordingly.")

if __name__ == "__main__":
    main()