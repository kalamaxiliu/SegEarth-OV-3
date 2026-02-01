_base_ = './base_config.py'

# === Inria Metadata ===
# 0=background, 1=building
metainfo = dict(
    classes=('background', 'building'),
    palette=[[0, 0, 0], [255, 0, 0]]
)

# === Model Settings ===
model = dict(
    type='SegEarthOV3Segmentation', # 显式指定类名，防止歧义
    # 类别文件
    classname_path='./configs/cls_inria.txt',
    gcm_alpha=1.0,  # 默认为 1.0。设为 2.0 可以放大抑制/增强的效果！
    # 推理参数
    confidence_threshold=0.35, # Inria 这种二分类可以适当放低阈值，靠 GCM 过滤
    prob_thd=0.35,
    slide_crop=512,  
    slide_stride=256,
    
    # === 关键创新点配置 (G-PCP) ===
    use_global_prior=True,        # 开启开关
    use_presence_score=True,      # 开启 Presence Head 调制
    
    # 指定 Inria 专属的先验文件
    prototype_path='weights/inria_prototypes.pkl',
    co_occurrence_path='data/co_occurrence_inria.json',
    dinov3_path='weights/dinov3/model.safetensors', # 确保此路径存在
    # ============================
)

# === Dataset Settings ===
dataset_type = 'BaseSegDataset'
data_root = 'data/Inria_Processed'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # reduce_zero_label=False 因为 Inria 0 就是背景，不是 ignore
    dict(type='LoadAnnotations', reduce_zero_label=False), 
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
        metainfo=metainfo, 
        img_suffix='.tif', 
        seg_map_suffix='.tif',
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])