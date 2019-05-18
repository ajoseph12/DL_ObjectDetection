import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from . import data_transforms


class ColorPatch_Dataset(Dataset):
    
   
    def __init__(self, path, img_size, is_training, is_debug=False):
        

        self.img_files = sorted([os.path.join(os.path.realpath('.'), path ,img) 
            for img in os.listdir(path) if img.endswith(".PNG")])
        self.label_files = sorted([os.path.join(os.path.realpath('.'), path, img)
             for img in os.listdir(path) if img.endswith(".txt")])
        
        logging.info("Total images: {}".format(len(self.img_files)))
        
        self.img_size = img_size  # (w, h)
        self.max_objects = 50
        self.is_debug = is_debug

        ## Transforms and augmentation
        self.transforms = data_transforms.Compose()
        
        #if is_training:
            #self.transforms.add(data_transforms.ImageBaseAug())
    
        #self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(data_transforms.ResizeImage(self.img_size))
        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))

    def __getitem__(self, index):

        ## Read images from list of image paths
        img_path = self.img_files[index % len(self.img_files)]
        
        img = self.__read_image(img_path)
	#img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        
        ori_h, ori_w = img.shape[:2]

        ## Read labels from list of label paths
        label_path = self.label_files[index % len(self.img_files)]
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            logging.info("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': labels}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample


    def __len__(self):
        return len(self.img_files)


    def __read_image(self, image_path):
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


#  use for test dataloader
if __name__ == "__main__":
    dataloader = torch.utils.data.DataLoader(ColorPatch_Dataset("../data/obj",
                                                         (608, 608), True, is_debug=True),
                                             batch_size=16,
                                             shuffle=False, num_workers=0, pin_memory=False)

    for step, sample in enumerate(dataloader):
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                	continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            plt.imsave("demo/step{}_{}.jpg".format(step, i),image)
            #cv2.imwrite("step{}_{}.jpg".format(step, i), image)
        # only one batch
        break
