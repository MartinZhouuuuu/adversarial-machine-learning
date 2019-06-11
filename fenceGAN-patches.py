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



def build_discriminator(image_shape):
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
	combined_model.compile(loss = combined_loss(fake_result,1,2),
		optimizer = 'adam')
	return combined_model


def train(GAN, G, D, epochs):
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
	
	noise = np.random.normal(0.5,0.5,(20000,latent_dim))
	d_loss_array = np.empty((0,1))
	g_loss_array = np.empty((0,1))
	fake_generated_2 = G.predict(noise)
	validity_g = D.predict(fake_generated_2)
	print(validity_g)
		
	for epoch in range(epochs):
		#train discriminator using fit_generator
		history = D.fit_generator(
		train_generator,
		steps_per_epoch = 50,
		epochs = 1
		)
		d_loss_1 = history.history['loss'][0]

		fake_generated_1 = G.predict(noise)
		fake_label_for_d = np.zeros(20000)
		d_loss_2 = D.train_on_batch(fake_generated_1,fake_label_for_d)
		
		fake_generated_2 = G.predict(noise)
		validity_g = D.predict(fake_generated_2)
		print(validity_g)
		
		d_loss = 0.5*d_loss_1 +0.5*d_loss_2
		d_loss_array = np.append(d_loss_array,np.array([[d_loss]]),axis=0)
		
		fake_label_for_g = np.zeros(20000)
		fake_label_for_g[:] = 0.5
		#train generator using train_on_batch
		g_loss = GAN.train_on_batch(noise,fake_label_for_g)
		
		#keep a record of generator loss
		g_loss_array = np.append(g_loss_array,np.array([[g_loss]]),axis = 0)
		fake_generated_2 = G.predict(noise)
		validity_g = D.predict(fake_generated_2)
		print(validity_g)
		
		#validity_d = D.predict_generator(train_generator,steps = 50)
		#print(validity_d)

	'''for x in range(fake_generated.shape[0]):
		plt.imshow(fake_generated[x].reshape(3,3),cmap = 'gray')
		#plt.imshow(fake_generated[0])
		plt.savefig('image/%d.png'%x)'''

G = build_generator(latent_dim,image_shape)
D = build_discriminator(image_shape)
GAN = fenceGAN(G,D)
train(GAN, G, D,5)
K.clear_session()