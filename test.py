import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img
import tifffile
import glob
import cv2

def get_dataset(path):
	dataset = np.empty((0,28,28,1))
	images = [tifffile.imread(file) for file in glob.glob(path)]
	for img in images:
		img_array = img_to_array(img)
		dataset = np.append(dataset,[img_array],axis = 0)
		
	return dataset

adversarial_dataset = get_dataset('full-fgsm/train/adversarial/*.tif')
original_dataset = get_dataset('full-fgsm/train/original/*.tif')
trainset = np.append(adversarial_dataset,original_dataset,axis = 0)
print(trainset.shape)