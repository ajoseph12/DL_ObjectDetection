# Dependencies
import glob, os
import cv2
import numpy as np
from tqdm import tqdm


class Preproc_Yolo(object):


	def __init__(self, base_img_paths, base_txt_paths, target_txt_path, target_train_path, test = False):


		self.target_txt_path = target_txt_path
		self.target_train_path = target_train_path

		if test:
			
			self.txt_paths = self._path_extractor(base_txt_paths, type = '.txt')
			self.roi, self.x_coords, self.y_coords = self._txt_treat()

			self.img_paths = self._path_extractor(base_img_paths, type = '.PNG')
			self.images = [cv2.imread(img_path) for img_path in tqdm(self.img_paths)]
			

		else:

			self.img_paths = sorted([os.path.join(os.path.realpath('.'), base_img_paths ,img) 
				for img in os.listdir(base_img_paths) if img.endswith(".PNG")])
			self.txt_paths = [img[:-4] + '_mask.txt' for img in self.img_paths]
			self.roi, self.x_coords, self.y_coords = self._txt_treat()
			self.images = [cv2.imread(img_path) for img_path in self.img_paths]

		self._creat_txt_files()
		self._create_train_txt()

		
	
	def _path_extractor(self, path, type):

		"""
		This function extracts names of text/image files and stores them in a list 
		and returns it
		"""

		file_paths = list()
		if type == '.txt':
			path_txt = path + '*mask.txt'
			for infile in sorted(glob.glob(path_txt)): file_paths.append(infile)

		elif type == '.PNG':
			path_img = path + '*.PNG'
			for infile in sorted(glob.glob(path_img)): file_paths.append(infile)


		return file_paths


	def _txt_treat(self):
		
		"""
		This method focuses on extracting the roi and xy coordinates from the .txt
		file of each corresponding image

		"""

		# store txt file content in list 'mask'
		mask = list()
		for i in self.txt_paths: 
			f =open(i, "r")
			mask.append(f.readlines())

		# convert string to numeric
		for i in range(len(mask)):
			for j in range(len(mask[i])):
				temp = [float(mask[i][j].strip().split(',')[k]) for k in range(4)] 
				mask[i][j] = temp


		# save values in lists: ROI, X and Y axis
		roi = [mask[i][0] for i in range(len(mask))]
		for i in range(len(roi)):
			roi[i] = [roi[i][0] + roi[i][2]/2, roi[i][1] + roi[i][3]/2, roi[i][2], roi[i][3]]

		x_coords = list()
		y_coords = list()
		
		for i in range(len(mask)):
			temp_x_coords = list()
			temp_y_coords = list()

			for j in range(1,len(mask[i]),2):
				temp_x_coords.append(mask[i][j])
				temp_y_coords.append(mask[i][j+1])

			x_coords.append(temp_x_coords)
			y_coords.append(temp_y_coords)

		return roi, x_coords, y_coords


	def _creat_txt_files(self):

		"""
		Onto creating text files at "self.target_path"

		roi class = 0
		color pixel classes = 1-24
		"""

		for i, image in enumerate(self.images):

			try: 
				h, w, _ = image.shape
				temp_label_box = '0'
				temp = self.standardize_box(h, w, box = self.roi[i])

				for j in temp: temp_label_box += ' '+str(j) 

				temp_label_box += '\n'
				x_i = self.x_coords[i]
				y_i = self.y_coords[i]

				for j in range(len(x_i)):

					temp_label_box += str(j+1)
					x_mid = (x_i[j][0] + x_i[j][3])/2 + (self.roi[i][0] - self.roi[i][2]/2)
					y_mid = (y_i[j][0] + y_i[j][1])/2 + (self.roi[i][1] - self.roi[i][3]/2)
					ht_temp = y_i[j][1] - y_i[j][0]
					wd_temp = x_i[j][3] - x_i[j][0]
					temp_box = [x_mid, y_mid, wd_temp, ht_temp]
					temp = self.standardize_box(h, w, temp_box)

					for k in temp: temp_label_box += ' '+str(k)
					temp_label_box += '\n'


				with open(self.target_txt_path + self.img_paths[i].split('/')[-1][:-4] + '.txt', 'w') as f:
					f.write(temp_label_box)
			
			except AttributeError:
				continue



	def _create_train_txt(self, base_path = 'data/obj/'):

		"""
		Creates a train.txt file with locations of each image 
		"""

		image_paths = ''
		for i in range(len(self.images)): 
			image_paths +=  base_path + self.img_paths[i].split('/')[-1] + '\n'

		with open(self.target_train_path + 'train.txt' , 'w') as f:
				f.write(image_paths)

	

	def standardize_box(self, h, w, box):

		"""
		Method to standardize box coordinates
		"""

		dw = 1/w
		dh = 1/h
		x = box[0]*dw
		y = box[1]*dh
		w = box[2]*dw
		h = box[3]*dh

		return x,y,w,h



def main(base_img_paths, base_txt_paths, target_txt_path, target_train_path):

	yolo = Preproc_Yolo(base_img_paths, base_txt_paths, target_txt_path, target_train_path)


if __name__ == '__main__':

	base_img_paths = 'Data/All/'
	base_txt_paths = 'Data/Nikon_D40/CHECKER/'
	target_txt_path = 'Data/PNG_TREATED/' # path where the txt files with box info is stored
	target_train_path = 'Data/' # path where the train.txt file pointing to images location is stored

	main(base_img_paths, base_txt_paths, target_txt_path, target_train_path)

