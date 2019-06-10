import cv2
import glob
import os
'''
Crops small patches from an image
Arguments:
	path: the file path from which images are read in
	filter_size: a tuple with 2 elements
	stride: an int 
'''
def crop_images(path,filter_size,stride):
	#read in images
	images = [cv2.imread(file) for file in glob.glob(path)]
	patch_count = 0
	for image in images:

		
		#stick to one row first
		for vertical_step in range((image.shape[0]-filter_size[0])//stride+1):
			
			#filter move down the row with a given stride
			for horizontal_step in range((image.shape[1]-filter_size[1])//stride+1):
				
				#picking the start and end points
				horizontal_start_point = 0 + stride*vertical_step
				horizontal_end_point = filter_size[0] + stride*vertical_step
				vertical_start_point = 0+stride*horizontal_step
				vertical_end_point = filter_size[1]+stride*horizontal_step
				
				#array slicing
				crop_image = image[horizontal_start_point:horizontal_end_point,
				vertical_start_point:vertical_end_point]
				
				#write image to a directory
				cv2.imwrite('patches/clean/%d.jpg'%
					(patch_count),
					crop_image)

				patch_count += 1

crop_images('350-MNIST-images/*.jpg',(3,3),1)