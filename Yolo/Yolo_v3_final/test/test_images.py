# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import torch.nn as nn


sys.path.append("..")
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 26)]

def read_image(image_path):
    """
    We read the image located at the path image_path

    Args:
        image_path (str): The path of the image
    
    Returns:
        np.ndarray: The image in numpy array
    """
    # We read the image
    try:
        image = plt.imread(image_path)
    except OSError as err:
        print("Error: " + str(err))
        return None

    # We take the maximum pixel value of each filter
    max = np.max(image, axis=0)
    max = np.max(max, axis=0, keepdims=True)
    max = np.tile(max, (image.shape[0], image.shape[1], 1))

    # We take the minimum pixel value of each filter
    min = np.min(image, axis=0)
    min = np.min(min, axis=0, keepdims=True)
    min = np.tile(min, (image.shape[0], image.shape[1], 1))

    # and we do a min-max normalization
    image = ((image - min) / (max-min))**(1/2.2)

    return image



def test(config):
    is_training = False
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # prepare images path
    images_name = os.listdir(config["images_path"])
    images_path = [os.path.join(config["images_path"], name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["images_path"]))

    # Start inference
    batch_size = config["batch_size"]
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))
            image = read_image(path)
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]),
                               interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32)
            #image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        
        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                   conf_thres=config["confidence_threshold"],
                                                   nms_thres=0.1)

        # write result images. Draw bounding boxes and labels of detections
        classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
        if not os.path.isdir("./output/"):
            os.makedirs("./output/")
        for idx, detections in enumerate(batch_detections):
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(images_origin[idx])
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config["img_h"], config["img_w"]
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                             edgecolor=color,
                                             facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
                             verticalalignment='top',
                             bbox={'color': color, 'pad': 0})
            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('output/{}_{}.jpg'.format(step, idx), bbox_inches='tight', pad_inches=0.0)
            plt.close()
    logging.info("Save all results to ./output/")    


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python test_images.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    test(config)


if __name__ == "__main__":
    main()
