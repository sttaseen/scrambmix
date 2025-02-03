model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dCSN',
        pretrained2d=False,
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
        depth=50,
        with_pool2=True,
        bottleneck_mode='ir',
        norm_eval=False,
        zero_init_residual=False,
        bn_frozen=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=52,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.68,
        init_std=0.01),
    # train_cfg=dict(blending=dict(type='Scrambmix', num_classes=52, num_frames=32, alpha=7.347)),
    test_cfg=dict(average_clips='prob', max_testing_views=10))

# This schedule is specifically for training on 8 GPUs with 1 images per GPU
checkpoint_config = dict(interval=5)

# Set up fold number
fold = 5

# Setup WandB
log_config = dict(interval=10,
                 hooks=[
                        dict(type='TextLoggerHook'),
                        dict(type='WandbLoggerHook',
                        init_kwargs={
                         'entity': "cares",
                         'project': "hmdb51-cv",
                         'group': f'baseline-fold{fold}',
                         'name': f'baseline-fold-{fold}'
                        },
                        log_artifact=True)
])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = f'data/hmdb51/annotation_train_fold_{fold}.txt'
ann_file_val = f'data/hmdb51/annotation_test_fold_{fold}.txt'
ann_file_test = f'data/hmdb51/annotation_test_fold_{fold}.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
gpu_ids = range(0, 1)
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=10,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    val_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file=ann_file_train,
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file=ann_file_val,
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file=ann_file_test,
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=10,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='ThreeCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
optimizer = dict(type='Adam', lr=7.886714129990479e-06, weight_decay=0.00001)
optimizer_config = dict(grad_clip=dict(max_norm=41, norm_type=2))
lr_config = None
total_epochs = 60
work_dir = f'./work_dirs/baseline-{fold}/'
find_unused_parameters = True
omnisource = False
module_hooks = []
