_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_potsdam.txt',
    # === 显式指定 JSON 路径 ===
    co_occurrence_path='data/co_occurrence_potsdam.json',
    prob_thd=0.1,
    confidence_threshold=0.2,
    bg_idx=5,
    gcm_alpha=2.4,  # 默认为 1.0。设为 2.0 可以放大抑制/增强的效果！
    # === 新增：必须开启滑动窗口推理 ===
    test_cfg=dict(
        mode='slide',           # 开启滑动模式
        crop_size=(1024, 1024), # 每次只推 1024x1024 的小块
        stride=(768, 768)       # 步长，保留重叠区域以保证边缘效果
    )
    # ===============================
)

# dataset settings
dataset_type = 'SegEarthPotsdamDataset'
data_root = 'data/Potsdam'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'), # 现在可以正常加载单通道 Mask 了
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 指定刚才脚本生成的 file_list
        ann_file='val.txt', 
        data_prefix=dict(
            img_path='Images',      # 原图文件夹
            seg_map_path='Labels_Index' # 转换后的单通道标签文件夹
        ),
        img_suffix='.tif',       # 原图后缀 (通常是 .tif)
        seg_map_suffix='.png',   # 转换后的 mask 后缀 (我们在脚本里存为了 .png)
        pipeline=test_pipeline))