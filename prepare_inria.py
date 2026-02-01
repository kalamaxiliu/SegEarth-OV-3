import os
import shutil
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# === 配置路径 ===
# 你的原始数据路径 (根据你的截图)
source_root = 'data/AerialImageDataset' 
# 目标路径 (MMSeg 标准格式)
target_root = 'data/Inria_Processed'

# === 配置要提取的验证集 ===
# Inria 包含的城市，每个城市取前 5 张做验证
cities = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
num_val_per_city = 5 

def prepare_data():
    # 1. 创建目录
    img_dest = os.path.join(target_root, 'img_dir', 'val')
    ann_dest = os.path.join(target_root, 'ann_dir', 'val')
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(ann_dest, exist_ok=True)

    print(f"正在从 {source_root} 提取数据到 {target_root} ...")

    # 2. 遍历城市并处理
    for city in cities:
        # 查找该城市的所有图片
        # 原始路径通常是 data/AerialImageDataset/train/images/*.tif
        search_path = os.path.join(source_root, 'train', 'images', f'{city}*.tif')
        files = sorted(glob(search_path))
        
        if len(files) == 0:
            print(f"警告: 在 {search_path} 没找到图片，请检查路径是否正确！")
            continue

        # 取前 N 张
        val_files = files[:num_val_per_city]
        
        for img_path in tqdm(val_files, desc=f"Processing {city}"):
            file_name = os.path.basename(img_path)
            
            # --- 处理图片 ---
            # 直接复制图片
            shutil.copy(img_path, os.path.join(img_dest, file_name))
            
            # --- 处理标签 ---
            # 对应的标签路径 (把 images 换成 gt)
            gt_path = img_path.replace('/images/', '/gt/')
            
            if not os.path.exists(gt_path):
                print(f"缺失标签: {gt_path}")
                continue
                
            # 读取标签 (Inria 标签是 0/255)
            # 使用 cv2 读取保持原始数值
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # 关键步骤：把 255 转成 1
            mask[mask > 0] = 1
            # mask[mask <= 127] = 0
            
            # 保存处理后的标签
            cv2.imwrite(os.path.join(ann_dest, file_name), mask)

    print("\n数据准备完成！")
    print(f"图片位置: {img_dest}")
    print(f"标签位置: {ann_dest}")

if __name__ == '__main__':
    prepare_data()
