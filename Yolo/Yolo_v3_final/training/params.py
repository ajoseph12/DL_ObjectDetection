TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 25,
    },
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.001,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 30,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 4,
    "train_path": "../data/PNG", #../data/color_sci/train.txt"
    "epochs": 90,
    "img_h": 608,
    "img_w": 608,
    "parallels": [0],                         #  config GPU device
    "working_dir": "model/", #  replace with your working dir
    "pretrain_snapshot": "",                        #  load checkpoint
    "evaluate_type": "", 
    "try": 0,
    "export_onnx": False,
}

#/Users/allwynjoseph/Desktop/MLDM master/Semester 3/dl_project/Yolo/Data/YOLO_V3/training/
#./data/coco/trainvalno5k.txt"
#../weights/darknet53_weights_pytorch.pth
