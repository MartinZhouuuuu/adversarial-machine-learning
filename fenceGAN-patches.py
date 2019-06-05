import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Reshape, Input, BatchNormalization
from keras.models import Sequential,Model
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
import tensorflow as tf
def crop_images(path,filter_size,stride):
	#read in images
	images = [cv2.imread(file) for file in glob.glob(path)]
	image_count = 0

	for image in images:
		#create one folder for patches from the same image
		if not os.path.exists(
			'patches/%d'%image_count
			):
			os.makedirs('patches/%d'%image_count)

		patch_count = 0
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
				cv2.imwrite('patches/%d/%d-(%d-%d)-(%d-%d).jpg'%
					(image_count,patch_count,
						vertical_start_point,vertical_end_point,
						horizontal_start_point,horizontal_end_point),
					crop_image)

				patch_count += 1
		image_count +=1

def generate_noise(latent_dim):
	noise = np.random.normal(0,1,[1000,latent_dim])
	return noise

def build_generator(latent_dim):
	model = Sequential()
	model.add(Dense(256,input_shape = (latent_dim,)))
	model.add(LeakyReLU(alpha = 0.2))
	model.add(BatchNormalization())

	model.add(Dense(512))
	model.add(LeakyReLU(alpha = 0.2))
	model.add(BatchNormalization())

	model.add(Dense(np.prod(image_shape)),activation = 'sigmoid')
	model.add(Reshape(image_shape))
	return model


def build_discrimintator(image_shape):
	model = Sequential()
	model.add(Flatten(input_shape = image_shape))
	model.add(Dense(512))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(256))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(128))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(1),activation =  'sigmoid')

	return model

def fenceGAN():
	image_rows = 32
	image_columns = 32
	image_channels = 3
	image_shape = (image_rows, image_columns, image_channels)
	latent_dim = 100
	discrimintator = build_discrimintator(image_shape)
	generator = build_generator(latent_dim)
	noise = generate_noise(latent_dim)
	discrimintator.compile(loss = discrimintator_loss, 
		optimizer = 'adam', metrics = ['accuracy'])

	z = Input(shape = (latent_dim,))
	fake_result = generator(z)
	discrimintator.trainable = False
	validity = discrimintator(fake_result)
	combined_model = Model(z,validity)
	combined_model.compile(loss = 'binary_crossentropy',
		optimizer = 'adam', metrics = ['accuracy'])



def discriminator_loss(y_true,y_generated):
	d_loss = binary_crossentropy(y_true,y_generated)
	#here might have to calculate weighted d_loss
	return d_loss

def generator_loss(y_true,y_generated):
	#gradient descent on alpha value
	encirclement_loss = binary_crossentropy(y_true,y_generated)
	dispersion_loss = 











