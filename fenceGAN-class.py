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
		self.batch_size = 64
		self.gm = 0.1
		self.k = 1
		self.fence_label = 0.5
		self.gamma = K.variable([1])
		self.g_optimizer = Adam(0.00002,0.5)
		self.d_optimizer = Adam(0.00002,0.5)
		self.G = self.build_generator()
		self.D = self.build_discriminator()
		self.GAN = self.combined_model()
		self.d_loss_array = np.empty((0,1))
		self.g_loss_array = np.empty((0,1))

		self.num_of_iterations = 10000

	def get_dataset(self,num_of_patches,path):
		dataset = np.empty((0,self.image_rows,self.image_columns,self.image_channels))
		filenames = []
		for file in os.listdir(path):
			if file.endswith('.tif'):
				filenames.append(file) 
		chosen_names = random.choices(filenames,k = num_of_patches)
		for name in chosen_names:
			image = tifffile.imread(os.path.join(path,name))
			dataset = np.append(dataset,[image],axis = 0)
		dataset = self.rescale(dataset)
		return dataset

	def build_generator(self):
		model = Sequential()
		model.add(Dense(7*7*128,input_dim = self.latent_dim))
		model.add(BatchNormalization())
		model.add(ReLU())
		model.add(Reshape((7,7,128)))
		model.add(Conv2DTranspose(64,(5,5), strides = 2, padding = 'same', kernel_initializer = 'glorot_normal'))
		model.add(BatchNormalization())
		model.add(ReLU())
		model.add(Conv2DTranspose(1,(5,5), strides = 2, padding = 'same', kernel_initializer = 'glorot_normal'))
		model.add(Activation('tanh'))
	
		model.summary()
		return model

	def rescale(self,images):
		images = images*2 -1
		return images

	def weighted_d_loss(self,y_true,y_predicted):
		d_loss = binary_crossentropy(y_true,y_predicted)
		d_loss_gen = d_loss*self.gamma
		
		return d_loss_gen

	def build_discriminator(self):
		model = Sequential()

		model.add(Conv2D(32,(3,3),strides = 2,padding = 'same',input_shape = self.image_shape))
		model.add(LeakyReLU(alpha = 0.2))
		
		model.add(Conv2D(64,(3,3),strides = 2,padding = 'same'))
		# model.add(BatchNormalization())
		model.add(LeakyReLU(alpha = 0.2))
		

		model.add(Conv2D(128,(3,3),strides = 2,padding = 'same'))
		# model.add(BatchNormalization())
		model.add(LeakyReLU(alpha = 0.2))
		

		model.add(Conv2D(256,(3,3),strides = 2,padding = 'same'))
		# model.add(BatchNormalization())
		model.add(LeakyReLU(alpha = 0.2))
		

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
		# 
		return combined_model

	def pretrain(self):
		for iteration in range(100):
			#get batch of clean patches
			batch_real = self.get_dataset(self.batch_size,'full-fgsm/train/original')
			real_label = np.ones(self.batch_size)

			#get batch of noise
			batch_noise_d = np.random.normal(0,1,(self.batch_size,self.latent_dim))
			noise_label = np.zeros(self.batch_size)
			
			self.D.trainable = True

			K.set_value(self.gamma,[1])
			
			fake_generated = self.G.predict(batch_noise_d)

			d_loss_1 = self.D.train_on_batch(fake_generated,noise_label)
			
			d_loss_2 = self.D.train_on_batch(batch_real,real_label)
			
			#record discriminator loss
			d_loss = 0.5 * np.add(d_loss_1,d_loss_2)
			print('iteration%d d_loss:%f'%(iteration,d_loss))

	def train(self):
		for iteration in range(self.num_of_iterations):
			##get batch of clean patches
			batch_real = self.get_dataset(self.batch_size,'full-fgsm/train/original')
			#get batch of noise
			batch_noise_d = np.random.normal(0,1,(self.batch_size,self.latent_dim))
			batch_noise_g = np.random.normal(0,1,(2*self.batch_size,self.latent_dim))
				
			#label for noise, clean patches and generator training
			real_label = np.ones(self.batch_size)
			noise_label = np.zeros(self.batch_size)
			half_label = np.zeros(2*self.batch_size)
			half_label[:] = self.fence_label
			iteration_d_loss = [0]
			for i in range(self.k):	
				#train discriminator
				K.set_value(self.gamma,[1])
				sub_iteration_d_loss_real = self.D.train_on_batch(batch_real,real_label)
				
				fake_generated = self.G.predict(batch_noise_d)
				K.set_value(self.gamma,[self.gm])
				sub_iteration_d_loss_fake = self.D.train_on_batch(fake_generated,noise_label)

				sub_iteration_d_loss = (1/self.k) * np.add(sub_iteration_d_loss_fake,sub_iteration_d_loss_real)
				iteration_d_loss = np.add(iteration_d_loss,sub_iteration_d_loss) 
			
			#train generator
			self.D.trainable = False
			iteration_g_loss = self.GAN.train_on_batch(batch_noise_g,half_label)
			print('Iteration%d D loss %0.5f G loss %0.5f'%(iteration, iteration_d_loss,iteration_g_loss))

			if iteration%50 == 0:
				self.progress_report(iteration)
				self.report_scores(iteration)
				self.save_model()
				
			self.g_loss_array = np.append(self.g_loss_array,np.array([[iteration_g_loss]]),axis = 0)
			self.d_loss_array = np.append(self.d_loss_array,np.array([[iteration_d_loss[0]]]),axis = 0)
			self.plot_losses()

	def report_scores(self,iteration):
		#get 1000 clean patches
		batch_real = self.get_dataset(1000,'full-fgsm/train/original')
		validity_d = self.D.predict(batch_real)
		
		#get 1000 noise
		batch_noise = np.random.normal(0,1,(1000,self.latent_dim))
		validity_g = self.GAN.predict(batch_noise)
		
		# get 1000 adv patches
		batch_adv = self.get_dataset(1000, 'full-fgsm/test/adversarial')
		validity_a = self.D.predict(batch_adv)
		
		sns.distplot(validity_d,hist = True,rug = False,label = 'real',kde = False)
		sns.distplot(validity_g,hist = True,rug = False,label = 'generated', kde = False)
		sns.distplot(validity_a,hist = True,rug = False,label = 'adversarial', kde = False)
		plt.legend(prop={'size': 14})
		plt.title('Density Plot of different patches')
		plt.xlabel('discriminator score')
		plt.ylabel('Density')
		plt.savefig('sample-scores/%d.png'%iteration)
		plt.close()
		
	def plot_losses(self):
		#plot the loss over epochs
		plt.plot(self.d_loss_array[:,0])
		plt.plot(self.g_loss_array[:,0])
		plt.ylabel('Loss')
		plt.xlabel('Iteration')
		plt.legend(['discriminator', 'generator'], loc='upper right')
		plt.savefig("plots/loss-fenceGAN.png")
		plt.close()
	
	def save_generated_images(self):
		noise = np.random.normal(0,1,(1000,self.latent_dim))
		generated_patches = self.G.predict(noise)
		for x in range(generated_patches.shape[0]):
			tifffile.imsave('generated-patches/%d.tif'%x,generated_patches[x])

	def save_model(self):
		self.G.save('model-files/G.h5')
		self.D.save('model-files/D.h5')


	def progress_report(self,iteration):
		row, column = 3,10
		
		batch_real = self.get_dataset(10,'full-fgsm/test/original')
		validity_d = self.D.predict(batch_real)
		
		batch_noise = np.random.normal(0,1,(10,self.latent_dim))
		batch_generated = self.G.predict(batch_noise)
		validity_g = self.GAN.predict(batch_noise)


		batch_adv = self.get_dataset(10,'full-fgsm/test/adversarial')
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
		fig.savefig('sample-images/%d.png'%iteration)
		plt.close()


fenceGAN = fenceGAN()
# fenceGAN.D = load_model('/Users/apple/Desktop/8350/D.h5',custom_objects = {'weighted_d_loss' = fenceGAN.weighted_d_loss})
# fenceGAN.progress_report(10000)
# fenceGAN.report_scores(10000)
# fenceGAN.pretrain()
fenceGAN.train()
# fenceGAN.save_model()









