import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import *
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
from keras.preprocessing.image import img_to_array
from custom_losses import *
import tifffile
import keras.backend as K
import random
import os
from keras.datasets import mnist
'''fenceGAN implementation
author@chengyang
'''

class fenceGAN():
	def __init__(self):
		self.image_rows = 28
		self.image_columns = 28
		self.image_channels = 1
		self.image_shape = (self.image_rows, self.image_columns, self.image_channels)
		self.latent_dim = 100
		self.batch_size = 128
		self.gm = 0.1
		self.gamma = K.variable([1])
		self.g_optimizer = Adam(0.0002,0.5)
		self.d_optimizer = Adam(0.0002,0.5)
		self.G = self.build_generator()
		self.D = self.build_discriminator()
		self.GAN = self.combined_model()

		self.d_loss_array = np.empty((0,1))
		self.g_loss_array = np.empty((0,1))

		self.num_of_patches = 60000
		self.num_of_adv = 5000
		self.num_of_iterations = self.num_of_patches//self.batch_size
		# self.clean_dataset = self.get_dataset(self.num_of_patches,'full-fgsm/test/original')
		(self.clean_dataset, _), (_, _) = mnist.load_data()
		self.clean_dataset = self.clean_dataset/127.5 -1
		self.clean_dataset = np.expand_dims(self.clean_dataset, axis = 3)
		
		
		self.real_indexes = np.random.randint(0,self.num_of_patches,10)
		self.adv_indexes = np.random.randint(0,self.num_of_adv,10)

	def get_dataset(self,num_of_patches,path):
		dataset = np.empty((0,28,28,1))
		filenames = os.listdir(path)
		chosen_names = random.choices(filenames,k = num_of_patches)
		for name in chosen_names:
			image = tifffile.imread(os.path.join(path,name))
			dataset = np.append(dataset,[image],axis = 0)
		
		return dataset

	def build_generator(self):
		model = Sequential()
		model.add(Dense(7*7*128,input_dim = self.latent_dim))
		model.add(ReLU())
		model.add(Reshape((7,7,128)))
		model.add(BatchNormalization())
		model.add(Conv2DTranspose(64,(5,5), strides = 2, padding = 'same', kernel_initializer = 'glorot_normal'))
		model.add(ReLU())
		model.add(BatchNormalization())
		model.add(Conv2DTranspose(1,(5,5), strides = 2, padding = 'same', kernel_initializer = 'glorot_normal'))
		model.add(Activation('tanh'))
	
		model.summary()
		return model


	def weighted_d_loss(self,y_true,y_predicted):
		d_loss = binary_crossentropy(y_true,y_predicted)
		d_loss_gen = d_loss*self.gamma
		
		return d_loss_gen

	def build_discriminator(self):
		model = Sequential()

		model.add(Conv2D(32,(3,3),strides = 2,padding = 'same',input_shape = self.image_shape))
		model.add(LeakyReLU(alpha = 0.2))
		model.add(BatchNormalization())
		
		model.add(Conv2D(64,(3,3),strides = 2,padding = 'same'))
		model.add(LeakyReLU(alpha = 0.2))
		model.add(BatchNormalization())

		model.add(Conv2D(128,(3,3),strides = 2,padding = 'same'))
		model.add(LeakyReLU(alpha = 0.2))
		model.add(BatchNormalization())

		model.add(Conv2D(256,(3,3),strides = 2,padding = 'same'))
		model.add(LeakyReLU(alpha = 0.2))
		model.add(BatchNormalization())

		model.add(GlobalAveragePooling2D())

		model.add(Dense(1,activation = 'sigmoid'))
		model.compile(loss = self.weighted_d_loss, 
			optimizer = self.d_optimizer)
		
		model.summary()
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
		for epoch in range(100):
			#get batch of clean patches
			real_index = np.random.randint(0,self.num_of_patches,self.batch_size)
			batch_real = self.clean_dataset[real_index]
			real_label = np.ones(self.batch_size)
			#get batch of noise
			batch_noise_d = np.random.normal(0,1,(self.batch_size,self.latent_dim))
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
				batch_noise_d = np.random.normal(0,1,(self.batch_size,self.latent_dim))
				batch_noise_g = np.random.normal(0,1,(2*self.batch_size,self.latent_dim))
				
				#label for noise, clean patches and generator training
				real_label = np.ones(self.batch_size)
				noise_label = np.zeros(self.batch_size)
				half_label = np.zeros(2*self.batch_size)
				half_label[:] = 1
				
				#train discriminator
				fake_generated_1 = self.G.predict(batch_noise_d)
				self.D.trainable = True
				K.set_value(self.gamma,[self.gm])
				iteration_d_loss_fake = self.D.train_on_batch(fake_generated_1,noise_label)
				
				K.set_value(self.gamma,[1])
				
				iteration_d_loss_real = self.D.train_on_batch(batch_real,real_label)

				
				iteration_d_loss = 0.5 * np.add(iteration_d_loss_fake,iteration_d_loss_real)


				#train generator
				self.D.trainable = False
				iteration_g_loss = self.GAN.train_on_batch(batch_noise_g,half_label)
				print('D loss %0.5f G loss %0.5f'%(iteration_d_loss,iteration_g_loss))
				epoch_d_loss += iteration_d_loss
				epoch_g_loss += iteration_g_loss
				if iteration%50 == 0:
					self.progress_report(iteration)
					self.report_scores(iteration)
			epoch_g_loss /= 500
			epoch_d_loss /= 50
			self.g_loss_array = np.append(self.g_loss_array,np.array([[epoch_g_loss]]),axis = 0)
			self.d_loss_array = np.append(self.d_loss_array,np.array([[epoch_d_loss]]),axis = 0)
			self.plot_losses()
			print('epoch%d d_loss:%f'%(epoch,epoch_d_loss))
			print('epoch%d g_loss:%f'%(epoch,epoch_g_loss))

	def report_scores(self,epoch):
		#get 1000 clean patches
		real_index = np.random.randint(0,self.num_of_patches,1000)
		batch_real = self.clean_dataset[real_index]
		validity_d = self.D.predict(batch_real)
		
		#get 1000 noise
		batch_noise = np.random.normal(0,1,(1000,self.latent_dim))
		validity_g = self.GAN.predict(batch_noise)
		
		#get 1000 adv patches
		batch_adv = self.get_dataset(1000, 'full-fgsm/test/adversarial')
		batch_adv = batch_adv*2 -1
		validity_a = self.D.predict(batch_adv)
		
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
		plt.savefig("plots/loss.png")
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
		
		batch_noise = np.random.normal(0,1,(10,self.latent_dim))
		batch_generated = self.G.predict(batch_noise)
		validity_g = self.GAN.predict(batch_noise)

		batch_adv = self.get_dataset(10, 'full-fgsm/test/adversarial')
		batch_adv = batch_adv*2 -1
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
fenceGAN.pretrain()
fenceGAN.train(100)
fenceGAN.save_model()
