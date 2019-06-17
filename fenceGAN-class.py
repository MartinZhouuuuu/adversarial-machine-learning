import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Reshape, Input, BatchNormalization,Flatten
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import img_to_array, array_to_img
from custom_losses import *
import tifffile
import seaborn

'''fenceGAN implementation
author@chengyang
'''
class fenceGAN():
	def __init__(self):
		self.image_rows = 5
		self.image_columns = 5
		self.image_channels = 1
		self.image_shape = (self.image_rows, self.image_columns, self.image_channels)
		self.latent_dim = 10
		self.batch_size = 32
		self.G = self.build_generator()
		self.D = self.build_discriminator()
		self.GAN = self.combined_model()
		self.d_loss_array = np.empty((0,1))
		self.g_loss_array = np.empty((0,1))
		self.num_of_patches = 57600
		self.num_of_noise = 2000
		self.num_of_adv = 5760
		self.clean_dataset = self.get_dataset(self.num_of_patches,'patches/clean/%d.tif')
		self.adv_dataset = self.get_dataset(self.num_of_adv,'patches-for-prediction/dirty-patches-test/%d.tif')
		self.noise = np.random.normal(0.5,0.5,(self.num_of_noise,self.latent_dim))

	def get_dataset(self, num_of_patches,path):
		dataset = np.empty((0,5,5,1))
		for patch_id in range(num_of_patches):
			img = tifffile.imread(path
				%patch_id)
			img_array = img_to_array(img)
			dataset = np.append(dataset,[img_array],axis = 0)
		return dataset

	def build_generator(self):
		model = Sequential()
		model.add(Dense(32,input_shape = (self.latent_dim,)))
		model.add(LeakyReLU(alpha = 0.4))
		model.add(BatchNormalization(momentum = 0.8))

		model.add(Dense(64))
		model.add(LeakyReLU(alpha = 0.4))
		model.add(BatchNormalization(momentum = 0.8))

		model.add(Dense(32))
		model.add(LeakyReLU(alpha = 0.4))
		model.add(BatchNormalization(momentum = 0.8))

		model.add(Dense(np.prod(self.image_shape)))
		model.add(LeakyReLU(alpha = 0.4))
		model.add(Reshape(self.image_shape))
		return model


	def build_discriminator(self):
		model = Sequential()
		model.add(Flatten(input_shape = self.image_shape))
		model.add(Dense(16))
		model.add(LeakyReLU(alpha = 0.4))

		model.add(Dense(8))
		model.add(LeakyReLU(alpha = 0.4))

		model.add(Dense(4))
		model.add(LeakyReLU(alpha = 0.4))

		model.add(Dense(2))
		model.add(LeakyReLU(alpha = 0.4))

		model.add(Dense(1,activation = 'sigmoid'))
		model.compile(loss = 'binary_crossentropy', 
			optimizer = 'rmsprop')
		return model

	def combined_model(self):
		#combined model
		z = Input(shape = (self.latent_dim,))
		fake_result = self.G(z)
		self.D.trainable = False
		validity = self.D(fake_result)
		combined_model = Model(z,validity)
		#here  the beta value is set to 15
		combined_model.compile(loss = combined_loss(fake_result,20,2),
			optimizer = 'rmsprop')
		self.D.trainable = True
		return combined_model
	def pretrain(self):
		for epoch in range(20):
			real_index = np.random.randint(0,self.num_of_patches,self.batch_size)
			batch_real = self.clean_dataset[real_index]
			#get batch of noise
			fake_index_for_d = np.random.randint(0,self.num_of_noise,self.batch_size)
			batch_noise_d = self.noise[fake_index_for_d]
			real_label = np.ones(self.batch_size)
			noise_label = np.zeros(self.batch_size)
			fake_generated_1 = self.G.predict(batch_noise_d)
			d_loss_1 = self.D.train_on_batch(fake_generated_1,noise_label)
			d_loss_2 = self.D.train_on_batch(batch_real,real_label)
			#record discriminator loss
			d_loss = 0.5*d_loss_1 + 0.5*d_loss_2
			print('epoch%d d_loss:%f'%(epoch,d_loss))

	def train(self, epochs):
		for epoch in range(epochs):
			for iteration in range(self.num_of_noise//self.batch_size):
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
				d_loss_1 = self.D.train_on_batch(fake_generated_1,noise_label)
				d_loss_2 = self.D.train_on_batch(batch_real,real_label)
				#record discriminator loss
				d_loss = 0.5*d_loss_1 + 0.5*d_loss_2
				
				self.d_loss_array = np.append(self.d_loss_array,np.array([[d_loss]]),axis=0)
				g_loss = 0
				#train generator 
				for k in range(2):
					g_loss_k = self.GAN.train_on_batch(batch_noise_g,half_label)
					g_loss += g_loss_k*0.5
				
			
				#record generator loss
				self.g_loss_array = np.append(self.g_loss_array,np.array([[g_loss]]),axis = 0)
			print('epoch%d d_loss:%f'%(epoch,d_loss))
			print('epoch%d g_loss:%f'%(epoch,g_loss))
			self.report_scores(epoch)
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
		adv_index = np.random.randint(0,self.num_of_adv,1000)
		batch_adv = self.adv_dataset[adv_index]
		validity_a = self.D.predict(batch_adv)
		
		sns.distplot(validity_d,hist = False,rug = True,label = 'real')
		sns.distplot(validity_g,hist = False,rug = True,label = 'generated')
		sns.distplot(validity_a,hist = False,rug = True,label = 'adversarial')
		plt.legend(prop={'size': 16})
		plt.title('Density Plot of different patches')
		plt.xlabel('discriminator score')
		plt.ylabel('Density')
		plt.savefig('loss-graph/%d.png'%epoch)
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
			plt.imshow(generated_patches[x].reshape(5,5),cmap = 'gray')
			plt.savefig('generated-patches/%d.tif'%x)
			plt.close()

	def save_model(self):
		self.G.save('model-files/G.h5')
		self.D.save('model-files/D.h5')

	'''def progress_report(self):
		self'''

		


fenceGAN = fenceGAN()
fenceGAN.pretrain()
fenceGAN.train(20)
fenceGAN.plot_losses()
# fenceGAN.save_model()
# fenceGAN.save_generated_images()
