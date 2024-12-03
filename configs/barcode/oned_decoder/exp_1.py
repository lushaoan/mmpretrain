expand_ratio = 4.0
token_mixer = {
    "type": "InceptionConcatDWConvTokenMixer",
    "res_scale": True,
    "square_kernel_size": 3,
    "dw_conv_w": 11,
    "dw_conv_h": 7,
}

arch_settings = [
    # num_blocks, channels, expand_ratio, stride_hw, token_mixer
    [1, 16, expand_ratio, (2, 2), token_mixer],
    [2, 24, expand_ratio, (2, 2), token_mixer],
    [3, 32, expand_ratio, (2, 2), token_mixer],
    [4, 64, expand_ratio, (1, 1), token_mixer],
    [3, 96, expand_ratio, (1, 1), token_mixer],
    [3, 160, expand_ratio, (1, 1), token_mixer],
]

# model settings
model = dict(
    type="OnedDecoder",
    backbone=dict(
        type="MultiStageBackbone", in_channels=3, arch_settings=arch_settings
    ),
    neck=dict(
        type="GlobalStripPooling",
        in_channels=160,
        out_channels=160,
        kernel_size=8,
        pool_axis="h",
    ),
    head=dict(
        type="CTCHead",
        in_channels=160,
        mid_channels=512,
        local_attn_expand_ratio=3,
        blank=0,
        num_classes=235,
        loss=dict(
            type="FocalCTCLoss",
            blank=0,
            alpha=0.0,
            gamma=0.0,
        ),
    ),
)

data_preprocessor = dict(
    type="OnedDecoderDataPreprocessor",
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=True,
)

bgr_mean = data_preprocessor["mean"][::-1]
bgr_std = data_preprocessor["std"][::-1]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 64), backend="cv2"),
    dict(type="OffsetAndPadIndices", offset=1, target_len=64, pad_value=0),
    dict(
        type="PackInputs",
        algorithm_keys=("indices", "corners", "indices_len"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 64), backend="cv2"),
    dict(type="OffsetAndPadIndices", offset=1, target_len=64, pad_value=0),
    dict(
        type="PackInputs",
        algorithm_keys=("indices", "corners", "indices_len"),
    ),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type="OnedDecoderDataset",
        dataset_path=[
            "/dataset/shaoanlu/dataset/tmp/gen128/",
            "/dataset/shaoanlu/dataset/tmp/gen39/",
            "/dataset/shaoanlu/dataset/tmp/gen93/",
            "/dataset/shaoanlu/dataset/tmp/genean/",
        ],
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type="OnedDecoderDataset",
        dataset_path=[
            "/dataset/shaoanlu/dataset/tmp/gen128/",
            "/dataset/shaoanlu/dataset/tmp/gen39/",
            "/dataset/shaoanlu/dataset/tmp/gen93/",
            "/dataset/shaoanlu/dataset/tmp/genean/",
        ],
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="OnedDecoderMetric")

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.01))
# learning policy
param_scheduler = dict(type="CosineAnnealingLR", by_epoch=True, T_max=1000)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=50)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)

# defaults to use registries in mmpretrain
default_scope = "mmpretrain"

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type="IterTimerHook"),
    # print log every 100 iterations.
    logger=dict(type="LoggerHook", interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type="ParamSchedulerHook"),
    # save checkpoint per epoch.
    checkpoint=dict(type="CheckpointHook", interval=50, max_keep_ckpts=2),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # validation results visualization, set True to enable it.
    visualization=dict(type="VisualizationHook", enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
