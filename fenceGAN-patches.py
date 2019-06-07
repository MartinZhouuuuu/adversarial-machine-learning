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
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
image_rows = 32
image_columns = 32
image_channels = 3
image_shape = (image_rows, image_columns, image_channels)
latent_dim = 100
batch_size = 32
def build_generator(latent_dim,image_shape):
	model = Sequential()
	model.add(Dense(256,input_shape = (latent_dim,)))
	model.add(LeakyReLU(alpha = 0.2))
	model.add(BatchNormalization())

	model.add(Dense(512))
	model.add(LeakyReLU(alpha = 0.2))
	model.add(BatchNormalization())

	model.add(Dense(np.prod(image_shape), activation = 'sigmoid'))
	model.add(Reshape(image_shape))
	return model



def build_discrimintator(image_shape):
	model = Sequential()
	model.add(Flatten(input_shape = image_shape))
	model.add(Dense(256))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(128))
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
		
		loss = encirclement_loss + beta*dispersion_loss
		return loss
	return generator_loss
def fenceGAN():
	discrimintator = build_discrimintator(image_shape)
	generator = build_generator(latent_dim,image_shape)

	z = Input(shape = (latent_dim,))
	fake_result = generator(z)
	discrimintator.trainable = False
	validity = discrimintator(fake_result)
	combined_model = Model(z,validity)
	combined_model.compile(loss = combined_loss(fake_result,15,2),
		optimizer = 'adam')
	return combined_model

def train(GAN, G, D, epochs, v_freq=10):
	train_datagen = ImageDataGenerator(
		rescale = 1./255
		)
	train_generator = train_datagen.flow_from_directory(
		'patches',
		target_size = (32,32),
		batch_size = 16,
		class_mode = 'binary',
		color_mode = 'rgb'
		)
	discrimintator = build_discrimintator(image_shape)
	generator = build_generator(latent_dim,image_shape)
	GAN = fenceGAN()

	noise = np.random.normal(0.5,0.5,(2000,100))
	for epoch in range(epochs):
		discrimintator.fit_generator(
		train_generator,
		steps_per_epoch = 50,
		epochs = 1
		)
		fake_label = np.zeros(2000)
		fake_label[:] = 0.5
		GAN.train_on_batch(noise,fake_label)

	
	fake_generated = generator.predict(noise)
	validity = GAN.predict(noise)
	print(validity)
	for x in range(10):
		plt.imshow(fake_generated[x])
		plt.savefig('image/%d.png'%x)

G = build_generator(latent_dim,image_shape)
D = build_discrimintator(image_shape)
GAN = fenceGAN()
train(GAN, G, D, 300)












