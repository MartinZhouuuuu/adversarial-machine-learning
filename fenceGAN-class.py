import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Reshape, Input, BatchNormalization,Flatten
from keras.models import Sequential,Model
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf
from keras import backend as K
from custom_losses import *
'''fenceGAN implementation
author@chengyang
'''
class fenceGAN():
	def __init__(self):
		self.image_rows = 5
		self.image_columns = 5
		self.image_channels = 1
		self.image_shape = (self.image_rows, self.image_columns, self.image_channels)
		self.latent_dim = 100
		self.batch_size = 32
		self.G = self.build_generator()
		self.D = self.build_discriminator()
		self.GAN = self.combined_model()
		self.d_loss_array = np.empty((0,1))
		self.g_loss_array = np.empty((0,1))
		self.num_of_patches = 57600
		self.num_of_noise = 2000
		self.clean_dataset = self.get_clean_dataset(self.num_of_patches)
		self.noise = np.random.normal(0.5,0.5,(self.num_of_noise,self.latent_dim))

	def get_clean_dataset(self, num_of_patches):
		dataset = np.empty((0,5,5,1))
		for patch_id in range(num_of_patches):
			img = load_img('/home/zhoucy/Documents/adversarial-machine-learning/patches/clean/%d.jpg'
				%patch_id,color_mode = 'grayscale')
			img_array = img_to_array(img)
			dataset = np.append(dataset,[img_array],axis = 0)
		return dataset

	def build_generator(self):
		model = Sequential()
		model.add(Dense(64,input_shape = (self.latent_dim,)))
		model.add(LeakyReLU(alpha = 0.2))
		
		model.add(Dense(32))
		model.add(LeakyReLU(alpha = 0.2))

		model.add(Dense(16))
		model.add(LeakyReLU(alpha = 0.2))

		model.add(Dense(np.prod(self.image_shape), activation = 'sigmoid'))
		model.add(Reshape(self.image_shape))
		return model


	def build_discriminator(self):
		model = Sequential()
		model.add(Flatten(input_shape = self.image_shape))
		model.add(Dense(16))
		model.add(LeakyReLU(alpha = 0.2))

		model.add(Dense(32))
		model.add(LeakyReLU(alpha = 0.2))

		model.add(Dense(1,activation = 'sigmoid'))
		model.compile(loss = 'binary_crossentropy', 
			optimizer = 'adam')
		return model

	def combined_model(self):
		#combined model
		z = Input(shape = (self.latent_dim,))
		fake_result = self.G(z)
		self.D.trainable = False
		validity = self.D(fake_result)
		combined_model = Model(z,validity)
		#here  the beta value is set to 15
		combined_model.compile(loss = combined_loss(fake_result,15,2),
			optimizer = 'adam')
		return combined_model


	def train(self, epochs):
		for epoch in range(epochs):
			for iteration in range(self.num_of_noise//self.batch_size):
				#get batch of clean patches
				real_index = np.random.randint(0,self.num_of_patches,self.batch_size)
				batch_real = self.clean_dataset[real_index]
				#get batch of noise
				fake_index = np.random.randint(0,self.num_of_noise,self.batch_size)
				batch_noise = self.noise[fake_index]
				
				#label for noise, clean patches and generator training
				real_label = np.ones(self.batch_size)
				noise_label = np.zeros(self.batch_size)
				half_label = np.zeros(self.batch_size)
				half_label[:] = 0.5
				
				#train discriminator
				fake_generated_1 = self.G.predict(batch_noise)
				d_loss_1 = self.D.train_on_batch(fake_generated_1,noise_label)
				d_loss_2 = self.D.train_on_batch(batch_real,real_label)
				#record discriminator loss
				d_loss = 0.5*d_loss_1 + 0.5*d_loss_2
				self.d_loss_array = np.append(self.d_loss_array,np.array([[d_loss]]),axis=0)

				#train generator 
				g_loss = self.GAN.train_on_batch(batch_noise,half_label)
				g_loss /=100
				#record generator loss
				self.g_loss_array = np.append(self.g_loss_array,np.array([[g_loss]]),axis = 0)


	def report_scores(self):
		#get batch of clean patches
		real_index = np.random.randint(0,self.num_of_patches,self.batch_size)
		batch_real = self.clean_dataset[real_index]
		
		validity_d = self.D.predict(batch_real)
		print(validity_d)
		
		#get batch of noise
		fake_index = np.random.randint(0,self.num_of_noise,self.batch_size)
		batch_noise = self.noise[fake_index]

		validity_g = self.GAN.predict(batch_noise)
		print(validity_g)

		
	def plot_losses(self):
		#plot the loss over epochs
		plt.plot(self.d_loss_array[:,0])
		plt.plot(self.g_loss_array[:,0])
		plt.ylabel('loss')
		plt.xlabel('iteration')
		plt.legend(['discriminator', 'generator'], loc='upper right')
		plt.savefig("images/loss.png")
		plt.close()
	
	def save_generated_images(self):
		generated_patches = self.G.predict(self.noise)
		for x in range(generated_patches.shape[0]):
			plt.imshow(generated_patches[x].reshape(5,5),cmap = 'gray')
			plt.savefig('generated-patches/%d.png'%x)
			plt.close()
fenceGAN = fenceGAN()
fenceGAN.train(5)
fenceGAN.report_scores()
fenceGAN.plot_losses()
fenceGAN.save_generated_images()
