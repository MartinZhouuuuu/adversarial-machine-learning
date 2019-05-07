import cv2
import glob

def crop_images(path,filter_size,stride):
	images = [cv2.imread(file) for file in glob.glob(path)]
	image_count = 0

	for image in images:
		patch_count = 0
		for vertical_step in range((image.shape[0]-filter_size[0])//stride+1):
			for horizontal_step in range((image.shape[1]-filter_size[1])//stride+1):
				crop_image = image[(0+stride*vertical_step):(filter_size[0]+stride*vertical_step),
				(0+stride*horizontal_step):(filter_size[1]+stride*horizontal_step)]
				cv2.imwrite('/Users/apple/Google Drive/adversarial-machine-learning/patches/%d-%d.jpg'%(image_count,patch_count),crop_image)
				patch_count += 1
		image_count +=1
crop_images('/Users/apple/Desktop/lca/*.jpg',(10,10),1)