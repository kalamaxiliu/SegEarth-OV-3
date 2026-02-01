_base_ = './base_config.py'

# === 关键修改：定义类别信息 ===
# Inria 是二分类：0=背景，1=建筑
metainfo = dict(
    classes=('background', 'building'),
    palette=[[0, 0, 0], [255, 0, 0]]  # 背景黑色，建筑红色
)

# 1. 模型设置
model = dict(
    classname_path='./configs/cls_inria.txt',
    confidence_threshold=0.4,
    prob_thd=0.4,
    slide_crop=512,  
    slide_stride=256
)

# 2. 数据集设置
dataset_type = 'BaseSegDataset'
data_root = 'data/Inria_Processed'

test_pipeline = [
    dict(type='LoadImageFromFile'),
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
        # === 关键修改：注入 metainfo ===
        metainfo=metainfo, 
        # ============================
        img_suffix='.tif', 
        seg_map_suffix='.tif',
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])