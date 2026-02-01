import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from mmseg.structures import SegDataSample
from segearthov3_segmentor import SegEarthOV3Segmentation

# 1. 设置路径
# 确保这里指向的是具体的 .pt 文件，不仅仅是目录
# 根据之前的上下文，你的权重文件名是 sam3.pt
checkpoint_path = '/root/user/cs_tcci_penghai/mch/SegEarth-OV-3/weights/sam3/sam3.pt'
img_path = 'resources/oem_koeln_50.tif'  # 确保这张图片存在，或者换成你自己的图片路径

# 检查模型文件是否存在
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"找不到模型文件，请检查路径: {checkpoint_path}")

# 2. 准备类别文件
name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')

# 3. 加载图片并预处理
if not os.path.exists(img_path):
    print(f"警告: 图片 {img_path} 不存在，将创建一个随机图片用于测试流程。")
    img = Image.new('RGB', (512, 512), color='white')
else:
    img = Image.open(img_path)

img_tensor = transforms.Compose([
    transforms.ToTensor(),
])(img).unsqueeze(0).to('cuda') 

data_sample = SegDataSample()
img_meta = {
    'img_path': img_path,
    'ori_shape': img.size
}
data_sample.set_metainfo(img_meta)

# 4. 初始化模型
print(f"正在加载模型: {checkpoint_path} ...")
model = SegEarthOV3Segmentation(
    type='SegEarthOV3Segmentation',
    model_type='SAM3',
    classname_path='./configs/my_name.txt',
    # === 关键修改: 传入模型权重路径 ===
    # 不同的代码实现参数名可能不同，通常是 'checkpoint' 或 'sam_checkpoint'
    # 如果代码报错说 unexpected argument，请尝试将 'checkpoint' 改为 'sam_checkpoint'
    checkpoint=checkpoint_path, 
    # =================================
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

# 5. 将模型移动到 GPU
# 这是必须的，因为你的 img_tensor 在 cuda 上
model.to('cuda') 

# 6. 推理
print("开始推理...")
# 注意：有些版本的 MM 代码 predict 接收 list，有些接收 tensor，这里保持原样
seg_pred = model.predict(img_tensor, data_samples=[data_sample])
seg_pred = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)

print("推理完成，正在保存结果...")

# 7. 可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(seg_pred, cmap='viridis')
ax[1].set_title("Segmentation Prediction")
ax[1].axis('off')

plt.tight_layout()
save_path = 'seg_pred.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"结果已保存至: {save_path}")