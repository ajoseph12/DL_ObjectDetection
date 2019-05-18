TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 25,
    },
    "batch_size": 16,
    "iou_thres": 0.5,
    "val_path": "../data/PNG",
    #"annotation_path": "../data/coco/annotations/instances_val2014.json",
    "img_h": 608,
    "img_w": 608,
    "parallels": [0],
    "pretrain_snapshot": "../weights/model_608.pth",
}
