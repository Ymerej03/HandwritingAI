import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os


def image_rotator(input_folder):
	"""
	image_rotator() rotates all the images in a given folder to increase the size of the training dataset.

	:param input_folder: the (relative/absolute) path to the folder containing the images that you want to "duplicate"
	"""
	# Get a list of all image files in the input folder
	image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	# Process each image in the input directory
	for image_file in image_files:
		input_path = os.path.join(input_folder, image_file)
		# for every image in the folder is randomly rotated by between -10 and 10 degrees. This is done to increase
		# the size of the training data and to mimic potential variability.
		for j in range(10):
			image = cv2.imread(input_path)
			name = image_file.split('.')[0]
			output_path = os.path.join(input_folder, f'{name}{j}.png')
			rows, cols, _ = image.shape
			m = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-10, 10), 1)
			dst = cv2.warpAffine(image, m, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
			cv2.imwrite(output_path, dst)
