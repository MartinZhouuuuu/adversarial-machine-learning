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
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K

image_rows = 3
image_columns = 3
image_channels = 1
image_shape = (image_rows, image_columns, image_channels)
latent_dim = 100
batch_size = 32


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



def build_discrimintator(image_shape):
	model = Sequential()
	model.add(Flatten(input_shape = image_shape))
	model.add(Dense(16))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(32))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(1,activation =  'sigmoid'))
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
		
		loss = 10*encirclement_loss + beta*dispersion_loss
		return loss
	return generator_loss





def fenceGAN():
	discrimintator = build_discrimintator(image_shape)
	generator = build_generator(latent_dim,image_shape)
	#combined model
	z = Input(shape = (latent_dim,))
	fake_result = generator(z)
	discrimintator.trainable = False
	validity = discrimintator(fake_result)
	combined_model = Model(z,validity)
	
	# here  the beta value is set to 15
	combined_model.compile(loss = combined_loss(fake_result,15,2),
		optimizer = 'adam')
	return combined_model

def train(GAN, G, D, epochs, v_freq=10):
	train_datagen = ImageDataGenerator(
		rescale = 1./255
		)
	train_generator = train_datagen.flow_from_directory(
		'patches',
		target_size = (3,3),
		batch_size = 32,
		class_mode = 'binary',
		color_mode = 'grayscale',
		classes = ['dirty','clean']
		)
	discrimintator = build_discrimintator(image_shape)
	generator = build_generator(latent_dim,image_shape)
	GAN = fenceGAN()
	
	noise = np.random.normal(0.5,0.5,(2000,latent_dim))
	d_loss = np.empty((0,1))
	g_loss = np.empty((0,1))

	for epoch in range(epochs):
		#train discriminator using fit_generator
		history = discrimintator.fit_generator(
		train_generator,
		steps_per_epoch = 50,
		epochs = 1
		)
		#keep a record of discriminator loss
		history_array = np.array([history.history['loss']])
		d_loss = np.append(d_loss,history_array,axis=0)
		
		fake_label = np.zeros(2000)
		fake_label[:] = 0.5
		#train generator using train_on_batch
		generator_loss = GAN.train_on_batch(noise,fake_label)
		print(generator_loss)
		#keep a record of generator loss
		g_loss_array = np.array([[generator_loss]])
		g_loss = np.append(g_loss,g_loss_array,axis = 0)
		fake_generated = generator.predict(noise)
		validity_g = discrimintator.predict(fake_generated)
		print(validity_g)
		validity_d = discrimintator.predict_generator(train_generator,steps = 50)
		print(validity_d)
	
	print(d_loss)
	print(g_loss)
	

	'''for x in range(fake_generated.shape[0]):
		plt.imshow(fake_generated[x].reshape(3,3),cmap = 'gray')
		#plt.imshow(fake_generated[0])
		plt.savefig('image/%d.png'%x)'''

G = build_generator(latent_dim,image_shape)
D = build_discrimintator(image_shape)
GAN = fenceGAN()
train(GAN, G, D,5)
K.clear_session()

