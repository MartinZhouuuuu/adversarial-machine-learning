import tifffile
import glob
import cv2
import random
'''
Crops small patches from an image
Arguments:
	path: the file path from which images are read in
	filter_size: a tuple with 2 elements
	stride: an int 
'''
def crop_images(path,filter_size,stride):
	#read in images
	images = [tifffile.imread(file) for file in glob.glob(path)]
	patch_count = 0
	for image in images:
		random_row = random.randint(0,22)
		random_column = random.randint(0,22)
		horizontal_start_point = random_column
		horizontal_end_point = horizontal_start_point + filter_size[1]
		vertical_start_point = random_row
		vertical_end_point = filter_size[0] + vertical_start_point
		#array slicing
		crop_image = image[horizontal_start_point:horizontal_end_point,
		vertical_start_point:vertical_end_point]
		
		#write image to a directory
		tifffile.imsave('patches/test/adversarial/%d.tif'%
			(patch_count),
			crop_image)

		patch_count += 1

crop_images('full-fgsm/test/adversarial/*.tif',(5,5),1)