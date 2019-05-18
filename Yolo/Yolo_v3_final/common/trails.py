
import os

import sys
sys.path.append("..") 

from nets.model_main import ModelMain 
from nets.yolo_loss import YOLOLoss

list_path = 'train.txt'
img_files = []
label_files = []
for path in open(list_path, 'r'):
	print(path)
	label_path = path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').strip()

	print(label_path)
	break