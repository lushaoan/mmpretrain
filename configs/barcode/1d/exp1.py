num_class = 3

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
dataset_type = "BarcodeDatasetLabelFolder"
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
    dict(type="Regularize"),
    dict(type="Resize", scale=(224, 64), backend="cv2"),
    dict(type="AutoContrast"),
    dict(type="Invert", prob=0.2),
    dict(type="GaussianBlur", magnitude_range=[-1, 1]),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Regularize"),
    dict(type="Resize", scale=(224, 64), backend="cv2"),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        img_file_paths=[
            # 最后的路径不要带 /
            # pos data
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1026/train/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1031/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1124/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_230403/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_230506/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_230511/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_231214/train/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/PDF417_data/pdf417_231106_1110/train/pdf417",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/PDF417_data/pdf417_231117/train/pdf417",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/PDF417_data/pdf417_231120/train/pdf417",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/train_pdf417_231108/pdf417",
            # neg data
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1026/train/neg",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1031/neg",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1124/neg",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_0104/neg",
        ],
        barcode_type="1d",
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        img_file_paths=[
            # 最后的路径不要带 /
            # pos data
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1026/train/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1031/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1124/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_230403/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_230506/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_230511/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_231214/train/bar",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/PDF417_data/pdf417_231106_1110/train/pdf417",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/PDF417_data/pdf417_231117/train/pdf417",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/PDF417_data/pdf417_231120/train/pdf417",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_datasets/precloc_oned_230522_nojitter/train_pdf417_231108/pdf417",
            # neg data
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1026/train/neg",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1031/neg",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_1124/neg",
            "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_cls/labeled/labeled_0104/neg",
        ],
        barcode_type="1d",
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
