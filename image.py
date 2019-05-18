import numpy as np
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from abc import ABC, abstractmethod
import pandas as pd
import cv2
from collections import defaultdict
import os
from tqdm import tqdm
plt.rcParams['figure.figsize'] = (20, 15)


class TextReaderMixIn(ABC):
    __slots__ = []

    @staticmethod
    def read_txt(path):
        return pd.read_csv(path, header=None)


class Box(object):

    def __init__(self, x1, x2, y1, y2):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __str__(self):
        return str([self.x1, self.x2, self.y1, self.y2])

    def scale(self, factor_x, factor_y):
        """
        Scale the box according to scaling factors
        """
        self.x1 = int(np.floor(self.x1 * factor_x))
        self.x2 = int(np.floor(self.x2 * factor_x))
        self.y1 = int(np.floor(self.y1 * factor_y))
        self.y2 = int(np.floor(self.y2 * factor_y))
        return self.x1, self.x2, self.y1, self.y2

    @property
    def plot(self):
        x = self.x1
        y = self.y1
        h = self.x2 - self.x1
        w = self.y2 - self.y1

        return [(x, y), h, w]


class BoxFactory(object):
    def __new__(cls, record):
        if isinstance(record, list):
            return Box(record[0], record[0]+record[2], record[1], record[1]+record[3])
        elif isinstance(record, dict):
            return Box(record['x'][0], record["x"][2],
                       record["y"][0], record["y"][2])


class Mask(TextReaderMixIn):

    def __init__(self, path):
        self.coordinates = self.read_txt(path)
        self.__init_boxes()
        self.index = 0

    def __setitem__(self, key, value):
        if key != "0":
            for i in range(4):
                value["x"][i] += self["0"].x1
            for j in range(4):
                value["y"][j] += self["0"].y1
        self.__dict__[key] = BoxFactory(value)

        return self

    def __getitem__(self, index):
        if isinstance(index, int) and index > 24:
            raise IndexError
        index = str(index)
        if hasattr(self, index):
            return self.__dict__[index]
        else:
            raise ValueError

    def __iter__(self):
        return self

    def __next__(self):
        try:
            next_box = self[self.index]
        except IndexError:
            self.index = 0
            raise StopIteration
        self.index += 1
        return next_box

    def __len__(self):
        return 25

    def __init_boxes(self):

        points = defaultdict()
        current_index = 0
        for index, row in self.coordinates.iterrows():
            if index == 0:
                points[index] = list(row)
                current_index += 1
            elif index % 2 != 0:
                points[current_index] = {'x': list(row)}
            else:
                points[current_index]['y'] = list(row)
                current_index += 1
        for key, value in points.items():
            self[str(key)] = value


class Picture(object):

    def __init__(self, image_path, mask_path=None, gamma=2.2):
        self.gamma = gamma
        self.image = cv2.imread(image_path, -1)

        if mask_path is not None:
            self.mask = Mask(mask_path)
        else:
            self.mask = None

    def scale(self):

        self.image = self.image / 4095
        self.image = (self.image ** 1.0/2.2)
        return self

    def show(self):
        _, ax = plt.subplots(1)
        ax.imshow(self.image)
        if self.mask is not None:
            for patch in self.mask:
                rect = Rectangle(*patch.plot, linewidth=1,
                                 edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()

    def reshape(self, target_size):

        factor_x = target_size / self.image.shape[1]
        factor_y = target_size / self.image.shape[0]
        self.image = cv2.resize(self.image, (target_size, target_size))
        if self.mask is not None:
            for box in self.mask:
                box.scale(factor_x, factor_y)
        return self


class ImageCreator(ABC):

    def __init__(self, image_path, mask_path, image_save_path,  output_size):

        self.image_path = image_path
        self.mask_path = mask_path
        self.image_save_path = image_save_path
        self.output_size = output_size

    @abstractmethod
    def create(self, num_images):
        raise NotImplementedError


class RetinaNetImageCreator(ImageCreator):

    def __init__(self, image_path, mask_path, image_save_path,  output_size):
        super(RetinaNetImageCreator, self).__init__(
            image_path, mask_path, image_save_path,  output_size)

    def create(self, num_images):
        images = [os.path.join(self.image_path, i)
                  for i in os.listdir(self.image_path)]
        masks = [os.path.join(self.mask_path, i)
                 for i in os.listdir(self.mask_path) if i.endswith("mask.txt")]

        if not os.path.isdir(self.image_save_path):
            os.mkdir(self.image_save_path)
        if num_images == -1:
            num_images = max(len(images), len(masks))

        with open(os.path.join(self.image_save_path, "meta.csv"), "w+") as csv:
            for i in tqdm(range(num_images)):
                image = Picture(images[i], masks[i]
                                ).scale().reshape(self.output_size)
                image_savename = "image{}.jpg".format(i)
                cv2.imwrite(os.path.join(
                    self.image_save_path, image_savename), image.image)
                for j, box in enumerate(image.mask):

                    output = ",".join(
                        [image_savename, str(box.x1), str(box.y1), str(box.x2), str(box.y2), str(j)])+"\n"
                    csv.write(output)
        with open(os.path.join(self.image_save_path, "classes.csv"), "w+") as csv:
            for i in range(25):
                csv.write("{},{}\n".format(i, i))


if __name__ == "__main__":

    import os
    path = "data/PNG"
    files = os.listdir(path)

    mask_path = "data/CHECKER"
    mask_files = os.listdir(mask_path)
    mask_files = [os.path.join(mask_path, i)
                  for i in mask_files if i.endswith("_mask.txt")]
    # pic = Picture(os.path.join(
    #    path, files[4]), mask_files[4]).scale().reshape(420).show()
    img = RetinaNetImageCreator(path, mask_path, "data/test", 420)
    img.create(5)
