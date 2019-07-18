import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Reshape, Input, BatchNormalization,Flatten, Conv2DTranspose
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam, rmsprop
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
from keras.preprocessing.image import img_to_array
from custom_losses import *
import tifffile
import keras.backend as K
import random
import os
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
		self.batch_size = 128
		self.gm = 0.1
		self.gamma = K.variable([1])
		# self.g_optimizer = Adam(lr = 1e-5,beta_1 = 0.5, beta_2 = 0.999, decay = 1e-5)
		# self.d_optimizer = Adam(lr = 3e-6,beta_1 = 0.5, beta_2 = 0.999, decay = 1e-5)
		self.g_optimizer = Adam(lr = 3e-5,beta_1 = 0.5, beta_2 = 0.999,decay = 1e-5)
		self.d_optimizer = Adam(lr = 1e-5,beta_1 = 0.5, beta_2 = 0.999,decay = 1e-5)
		self.G = self.build_generator()
		self.D = self.build_discriminator()
		self.GAN = self.combined_model()

		self.d_loss_array = np.empty((0,1))
		self.g_loss_array = np.empty((0,1))

		self.num_of_patches = 30000
		self.num_of_noise = 30000
		self.num_of_adv = 1000
		self.num_of_iterations = self.num_of_patches//self.batch_size
		self.clean_dataset = self.get_dataset(self.num_of_patches,'patches/train/original')
		self.clean_dataset = self.clean_dataset.astype('float32')
		
		self.adv_dataset = self.get_dataset(self.num_of_adv,'patches/train/adversarial')
		self.adv_dataset = self.adv_dataset.astype('float32')
		
		self.noise = np.random.normal(0.5,0.5,(self.num_of_noise,self.latent_dim))
		
		self.real_indexes = np.random.randint(0,self.num_of_patches,10)
		self.adv_indexes = np.random.randint(0,self.num_of_adv,10)

	def get_dataset(self,num_of_patches,path):
		dataset = np.empty((0,5,5,1))
		filenames = os.listdir(path)
		for name in filenames:
			image = tifffile.imread(os.path.join(path,name))
			dataset = np.append(dataset,[image],axis = 0)
		
		return dataset

	def build_generator(self):
		model = Sequential()
		
		model.add(Dense(3*3*512,input_shape = (self.latent_dim,),activation = 'relu'))
		model.add(Reshape(3,3,512))
		model.add(Conv2DTranspose(256,(3,3),stride = ,activation = 'relu'))
		

		return model


	def weighted_d_loss(self,y_true,y_predicted):
		d_loss = binary_crossentropy(y_true,y_predicted)
		d_loss_gen = d_loss*self.gamma
		
		return d_loss_gen

	def build_discriminator(self):
		model = Sequential()
		
		model.add(Flatten(input_shape = self.image_shape))

		model.add(Dense(64))
		model.add(LeakyReLU(alpha = 0.4))

		model.add(Dense(128))
		model.add(LeakyReLU(alpha = 0.4))

		model.add(Dense(256))
		model.add(LeakyReLU(alpha = 0.4))

		model.add(Dense(1,activation = 'sigmoid'))
		model.compile(loss = self.weighted_d_loss, 
			optimizer = self.d_optimizer)
		
		return model

	def combined_model(self):
		#combined model
		self.D.trainable = False
		z = Input(shape = (self.latent_dim,))
		fake_result = self.G(z)
		validity = self.D(fake_result)
		combined_model = Model(z,validity)
		combined_model.compile(loss = combined_loss(fake_result,15,2),
			optimizer = self.g_optimizer)
		
		return combined_model

	def pretrain(self):
		for epoch in range(20):
			#get batch of clean patches
			real_index = np.random.randint(0,self.num_of_patches,self.batch_size)
			batch_real = self.clean_dataset[real_index]
			real_label = np.ones(self.batch_size)
			#get batch of noise
			fake_index_for_d = np.random.randint(0,self.num_of_noise,self.batch_size)
			batch_noise_d = self.noise[fake_index_for_d]
			noise_label = np.zeros(self.batch_size)
			
			fake_generated_1 = self.G.predict(batch_noise_d)
			
			self.D.trainable = True
			K.set_value(self.gamma,[self.gm])
			d_loss_1 = self.D.train_on_batch(fake_generated_1,noise_label)
			K.set_value(self.gamma,[1])
			d_loss_2 = self.D.train_on_batch(batch_real,real_label)
			
			#record discriminator loss
			d_loss = 0.5*d_loss_1 + 0.5*d_loss_2
			print('epoch%d d_loss:%f'%(epoch,d_loss))

	def train(self, epochs):
		for epoch in range(epochs):
			epoch_d_loss = 0
			epoch_g_loss = 0
			for iteration in range(self.num_of_iterations):
				#get batch of clean patches
				real_index = np.random.randint(0,self.num_of_patches,self.batch_size)
				batch_real = self.clean_dataset[real_index]
				
				#get batch of noise
				fake_index_for_d = np.random.randint(0,self.num_of_noise,self.batch_size)
				batch_noise_d = self.noise[fake_index_for_d]
				fake_index_for_g = np.random.randint(0,self.num_of_noise,2*self.batch_size)
				batch_noise_g = self.noise[fake_index_for_g]
				
				#label for noise, clean patches and generator training
				real_label = np.ones(self.batch_size)
				noise_label = np.zeros(self.batch_size)
				half_label = np.zeros(2*self.batch_size)
				half_label[:] = 0.5
				
				#train discriminator
				fake_generated_1 = self.G.predict(batch_noise_d)
				self.D.trainable = True
				K.set_value(self.gamma,[self.gm])
				epoch_d_loss += self.D.train_on_batch(fake_generated_1,noise_label)
				K.set_value(self.gamma,[1])
				epoch_d_loss += self.D.train_on_batch(batch_real,real_label)
				
				#record discriminator loss
				# d_loss = 5*d_loss_1 + 5*d_loss_2
				
				
				#train generator
				self.D.trainable = False
				epoch_g_loss += self.GAN.train_on_batch(batch_noise_g,half_label)
				
				#record generator loss
			epoch_g_loss /= 500
			epoch_d_loss /= 50
			self.g_loss_array = np.append(self.g_loss_array,np.array([[epoch_g_loss]]),axis = 0)
			self.d_loss_array = np.append(self.d_loss_array,np.array([[epoch_d_loss]]),axis=0)
			self.plot_losses()
			print('epoch%d d_loss:%f'%(epoch,epoch_d_loss))
			print('epoch%d g_loss:%f'%(epoch,epoch_g_loss))
			self.report_scores(epoch)
			self.progress_report(epoch)

	def report_scores(self,epoch):
		#get 1000 clean patches
		real_index = np.random.randint(0,self.num_of_patches,1000)
		batch_real = self.clean_dataset[real_index]
		validity_d = self.D.predict(batch_real)
		
		#get 1000 noise
		fake_index = np.random.randint(0,self.num_of_noise,1000)
		batch_noise = self.noise[fake_index]
		validity_g = self.GAN.predict(batch_noise)
		
		#get 1000 adv patches
		validity_a = self.D.predict(self.adv_dataset)
		
		sns.distplot(validity_d,hist = True,rug = False,label = 'real')
		sns.distplot(validity_g,hist = True,rug = False,label = 'generated')
		sns.distplot(validity_a,hist = True,rug = False,label = 'adversarial')
		plt.legend(prop={'size': 14})
		plt.title('Density Plot of different patches')
		plt.xlabel('discriminator score')
		plt.ylabel('Density')
		plt.savefig('sample-scores/%d.png'%epoch)
		plt.close()
		
	def plot_losses(self):
		#plot the loss over epochs
		plt.plot(self.d_loss_array[:,0])
		plt.plot(self.g_loss_array[:,0])
		plt.ylabel('loss')
		plt.xlabel('iteration')
		plt.legend(['discriminator', 'generator'], loc='upper right')
		plt.savefig("loss-graph/loss.png")
		plt.close()
	
	def save_generated_images(self):
		generated_patches = self.G.predict(self.noise)
		for x in range(generated_patches.shape[0]):
			tifffile.imsave('generated-patches/%d.tif'%x,generated_patches[x])

	def save_model(self):
		self.G.save('model-files/G.h5')
		self.D.save('model-files/D.h5')

	def progress_report(self,epoch):
		row, column = 3,10
		
		batch_real = self.clean_dataset[self.real_indexes]
		validity_d = self.D.predict(batch_real)
		
		batch_noise = self.noise[:10]
		batch_generated = self.G.predict(batch_noise)
		validity_g = self.GAN.predict(batch_noise)

		batch_adv = self.adv_dataset[self.adv_indexes]
		validity_a = self.D.predict(batch_adv)
		
		fig, axs = plt.subplots(row,column)
		for j in range(column):
			axs[0,j].imshow(batch_real[j,:,:,0],cmap = 'gray')
			axs[0,j].axis('off')
			axs[0,j].set_title('%0.3f'%validity_d[j,:],size = 12)
			
			axs[1,j].imshow(batch_generated[j,:,:,0],cmap = 'gray')
			axs[1,j].axis('off')
			axs[1,j].set_title('%0.3f'%validity_g[j,:],size = 12)
			
			axs[2,j].imshow(batch_adv[j,:,:,0],cmap = 'gray')
			axs[2,j].axis('off')
			axs[2,j].set_title('%0.3f'%validity_a[j,:],size = 12)
		fig.text(0.45, 0.65, 'real',fontsize = 15)
		fig.text(0.45, 0.4, 'generated',fontsize = 15)
		fig.text(0.45, 0.1, 'adversarial',fontsize = 15)
		fig.savefig('sample-images/%d.png'%epoch)
		plt.close()

fenceGAN = fenceGAN()
# fenceGAN.pretrain()
fenceGAN.train(100)
fenceGAN.save_model()
fenceGAN.save_generated_images()
