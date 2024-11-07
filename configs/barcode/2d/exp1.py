num_class = 4

# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="MobileNetV3", arch="large"),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="StackedLinearClsHead",
        num_classes=num_class,
        in_channels=960,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type="HSwish"),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        init_cfg=dict(type="Normal", layer="Linear", mean=0.0, std=0.01, bias=0.0),
        topk=(1, 5),
    ),
)


# dataset settings
dataset_type = "BarcodeDataset"
data_preprocessor = dict(
    num_classes=num_class,
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
    dict(type="RandomResizedCrop", scale=224, backend="pillow"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(
        type="AutoAugment",
        policies="imagenet",
        hparams=dict(pad_val=[round(x) for x in bgr_mean]),
    ),
    dict(
        type="RandomErasing",
        erase_prob=0.2,
        mode="rand",
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std,
    ),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ResizeEdge", scale=256, edge="short", backend="pillow"),
    dict(type="CenterCrop", crop_size=224),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file_list=[
            # pos data
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/dm/train.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/mqr/train.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/qr/train.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/rect_dm/train.txt",
            },
            # neg data
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/dm/train.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/mqr/train.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/qr/train.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/rect_dm/train.txt",
            },
        ],
        barcode_type="2d",
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file_list=[
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/dm/test.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/mqr/test.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/qr/test.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/pos/rect_dm/test.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/dm/test.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/mqr/test.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/qr/test.txt",
            },
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902",
                "path": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/neg/rect_dm/test.txt",
            },
        ],
        barcode_type="2d",
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="Accuracy", topk=(1,))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
)
# learning policy
param_scheduler = dict(
    type="MultiStepLR", by_epoch=True, milestones=[100, 150], gamma=0.1
)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)
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
    checkpoint=dict(type="CheckpointHook", interval=40),
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
