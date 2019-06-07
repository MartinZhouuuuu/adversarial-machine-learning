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
image_rows = 32
image_columns = 32
image_channels = 3
image_shape = (image_rows, image_columns, image_channels)
latent_dim = 100

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

def build_generator(latent_dim):
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
	model.add(Dense(512))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(256))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(128))
	model.add(LeakyReLU(alpha = 0.2))

	model.add(Dense(1,activation =  'sigmoid'))
	model.compile(loss = com1(), 
		optimizer = 'adam')
	return model
def com1():
	def discriminator_loss(y_true,y_generated):
		d_loss = binary_crossentropy(y_true,y_generated)
		#here might have to calculate weighted d_loss
		return d_loss
	return discriminator_loss

def com2(G_out,gamma,power):
	def generator_loss(y_true,y_generated):
		#gradient descent on alpha value
		encirclement_loss = tf.reduce_mean(binary_crossentropy(y_true, y_generated))
		center = tf.reduce_mean(G_out, axis=0, keepdims=True)
		distance_xy = tf.pow(tf.abs(tf.subtract(G_out,center)),power)
		distance = tf.reduce_sum(distance_xy, 1)
		avg_distance = tf.reduce_mean(tf.pow(distance, 1/power))
		dispersion_loss = tf.reciprocal(avg_distance)
		
		loss = encirclement_loss + gamma*dispersion_loss
		return loss
	return generator_loss

def noise_data(n):
    return np.random.normal(0,8,n)

def data_G(batch_size):
    x = noise_data(batch_size)
    y = np.zeros(batch_size)
    y[:] = 0.5    #setting 1 from 'bullshit' data so that (G+D) can backprop to achieve this fake
    return x, y

def data_D(G,n_samples,mode):
    if mode == 'real':
        x = real_data(n_samples)
        y = np.ones(n_samples)
        return x, y
        
    elif mode == 'gen':
        x = G.predict(noise_data(n_samples))
        y = np.zeros(n_samples)
        return x, y

def fenceGAN():
	discrimintator = build_discrimintator(image_shape)
	generator = build_generator(latent_dim)

	z = Input(shape = (latent_dim,))
	fake_result = generator(z)
	discrimintator.trainable = False
	validity = discrimintator(fake_result)
	combined_model = Model(z,validity)
	combined_model.compile(loss = com2(fake_result,15,2),
		optimizer = 'adam')
	return combined_model

def train(GAN, G, D, epochs, n_samples, v_freq=10):
    d_loss = []
    g_loss = []

#    data_show = sample_noise(n_samples=n_samples)[0]
    for epoch in range(epochs):
        try:
            loss_temp = []
            x,y = data_D(G, n_samples, 'real')
            loss_temp.append(D.train_on_batch(x, y))
            
            set_trainability(D, True)
            K.set_value(gm, [gamma])
            x,y = data_D(G, n_samples, 'gen')
            loss_temp.append(D.train_on_batch(x,y))
            
            d_loss.append(sum(loss_temp)/len(loss_temp))
            
            ###Train Generator
            X, y = data_G(n_samples)
            generator_loss = GAN.train_on_batch(X,y)
            g_loss.append(generator_loss)
            
            if (epoch + 1) % v_freq == 0:
                print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
        except KeyboardInterrupt: #hit control-C to exit and save video there
            break
    return d_loss, g_loss
G = build_generator(latent_dim)
D = build_discrimintator(image_shape)
GAN = fenceGAN()
d_loss,g_loss = train(GAN, G, D, 3000, 128)












