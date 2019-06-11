import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Reshape, Input, BatchNormalization,Flatten
from keras.models import Sequential,Model
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import tensorflow as tf
from keras import backend as K

image_rows = 5
image_columns = 5
image_channels = 1
image_shape = (image_rows, image_columns, image_channels)
latent_dim = 100
batch_size = 32


def get_clean_dataset(num_of_patches):
	dataset = np.empty((0,5,5,1))
	for patch_id in range(num_of_patches):
		img = load_img('/home/zhoucy/Documents/adversarial-machine-learning/patches/clean/%d.jpg'%patch_id,
			color_mode = 'grayscale')
		img_array = img_to_array(img)
		dataset = np.append(dataset,[img_array],axis = 0)
	return dataset


def build_generator(latent_dim,image_shape):
	model = Sequential()
	model.add(Dense(64,input_shape = (latent_dim,)))
	model.add(LeakyReLU(alpha = 0.2))
	
	model.add(Dense(32))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(16))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(np.prod(image_shape), activation = 'sigmoid'))
	model.add(Reshape(image_shape))
	return model



def build_discriminator(image_shape):
	model = Sequential()
	model.add(Flatten(input_shape = image_shape))
	model.add(Dense(16))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(32))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(1,activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', 
		optimizer = 'adam')
	return model

def combined_loss(generated,beta,power):
	def generator_loss(y_true,y_generated):
		#gradient descent on alpha value
		encirclement_loss = tf.reduce_mean(binary_crossentropy(y_true, y_generated))
		center = tf.reduce_mean(generated, axis=0, keepdims=True)
		distance_xy = tf.pow(tf.abs(tf.subtract(generated,center)),power)
		distance = tf.reduce_sum(distance_xy, 1)
		avg_distance = tf.reduce_mean(tf.pow(distance, 1/power))
		dispersion_loss = tf.reciprocal(avg_distance)
		
		loss = encirclement_loss + beta*dispersion_loss
		return loss
	return generator_loss


def fenceGAN(G,D):
	#combined model
	z = Input(shape = (latent_dim,))
	fake_result = G(z)
	D.trainable = False
	validity = D(fake_result)
	combined_model = Model(z,validity)
	# here  the beta value is set to 15
	combined_model.compile(loss = combined_loss(fake_result,15,2),
		optimizer = 'adam')
	return combined_model


def train(GAN, G, D, epochs):
	clean_dataset = get_clean_dataset(57600)
	noise = np.random.normal(0.5,0.5,(2000,latent_dim))
	d_loss_array = np.empty((0,1))
	g_loss_array = np.empty((0,1))

	for epoch in range(epochs):
		for iteration in range(57600//32):
			real_index = np.random.randint(0,57600,32)
			fake_index = np.random.randint(0,2000,32)
			
			#train discriminator
			batch_noise = noise[fake_index]
			noise_label = np.zeros(32)
			batch_real = clean_dataset[real_index]
			real_label = np.ones(32)
			fake_generated_1 = G.predict(batch_noise)
			d_loss_1 = D.train_on_batch(fake_generated_1,noise_label)
			d_loss_2 = D.train_on_batch(batch_real,real_label)
		
			d_loss = 0.5*d_loss_1 + 0.5*d_loss_2
			d_loss_array = np.append(d_loss_array,np.array([[d_loss]]),axis=0)
		
			for k in range(10):
				half_label = np.zeros(32)
				half_label[:] = 0.5
				#train generator 
				g_loss = GAN.train_on_batch(batch_noise,half_label)
			
				#keep a record of generator loss
				g_loss_array = np.append(g_loss_array,np.array([[g_loss]]),axis = 0)
		
		print('d_loss: %f'%d_loss)
		print('g_loss: %f'%g_loss)

		fake_generated_2 = G.predict(batch_noise)
		validity_g = D.predict(fake_generated_2)
		print(validity_g)
		
		validity_d = D.predict(batch_real)
		print(validity_d)
	print(d_loss_array)
	print(g_loss_array)
	'''for x in range(fake_generated_2.shape[0]):
		plt.imshow(fake_generated_2[x].reshape(3,3),cmap = 'gray')
		plt.savefig('image/%d.png'%x)'''





G = build_generator(latent_dim,image_shape)
D = build_discriminator(image_shape)
GAN = fenceGAN(G,D)
train(GAN, G, D,5)
K.clear_session()