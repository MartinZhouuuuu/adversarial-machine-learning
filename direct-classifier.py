'''this is a trial on MNIST dataset
which is to directly classify clean and dirty images
unlikely to succeed'''
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD, Adam, rmsprop
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import to_categorical
import tifffile
import keras.backend as K
import os
import random
class Clean_dirty_classifier():
	def __init__(self):
		self.num_of_columns = 28
		self.num_of_rows = 28
		self.image_shape = (self.num_of_rows,self.num_of_columns,1)
		self.optimizer = Adam(lr = 0.0003)
		self.classifier = self.classifier()
		self.train_loss_array = np.empty((0,1))
		self.train_acc_array = np.empty((0,1))

	def get_dataset(self,path,sample_size):
		dataset = np.empty((0,28,28,1))
		labels = np.empty((0,2))
		for folder in os.listdir(path):	
			if folder == 'original':
				filenames = os.listdir(os.path.join(path,folder))
				chosen_names = random.choices(filenames,k = sample_size)
				for name in chosen_names:
					image = tifffile.imread(os.path.join(path,folder,name))
					dataset = np.append(dataset,[image],axis = 0)

				ones = 	to_categorical(np.ones(sample_size),num_classes = 2)
				labels = np.append(labels,ones,axis = 0)

			elif folder == 'adversarial':
				filenames = os.listdir(os.path.join(path,folder))
				chosen_names = random.choices(filenames,k = sample_size)
				for name in chosen_names:
					image = tifffile.imread(os.path.join(path,folder,name))
					dataset = np.append(dataset,[image],axis = 0)
				zeros = to_categorical(np.zeros(sample_size),num_classes = 2)
				labels = np.append(labels,zeros,axis = 0)
		return dataset, labels

	def classifier(self):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding='same',
							 input_shape= self.image_shape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Conv2D(64, (3, 3), padding = 'same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Conv2D(128, (3, 3), padding = 'same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Flatten())
		model.add(Dense(32))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		model.add(Dense(2))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy',
					  optimizer= self.optimizer,
					  metrics=['accuracy'])
		return model

	def train(self,epochs):
		for epoch in range(epochs):
			epoch_loss = 0
			epoch_acc = 0
			for iteration in range(100):
				train_x, train_y = self.get_dataset(
					'/Users/apple/Google Drive/adversarial-machine-learning/full-fgsm/train',
					32
					)
				train_return = self.classifier.train_on_batch(train_x,train_y)
				epoch_loss += train_return[0]
				epoch_acc += train_return[1]


		epoch_loss /= 100
		epoch_acc /= 100
		print('Epoch%d train_loss:%0.3f train_acc:%0.3f'%(epoch, epoch_loss,epoch_acc))
		self.train_loss_array = np.append(self.train_loss_array,np.array([[epoch_loss]]),axis=0)
		self.train_acc_array = np.append(self.train_acc_array,np.array([[epoch_acc]]),axis=0)
		self.save_model()


	def save_model(self):
		self.classifier.save('model-files/classifier.h5')


	def test(self):
		test_x, test_y = self.get_dataset(
					'/Users/apple/Google Drive/adversarial-machine-learning/full-fgsm/test',
					500
					)
		best_model = load_model('model-files/classifier.h5')
		test_loss,test_acc = best_model.evaluate(test_x,test_y)
		print('test_loss:%0.3f test_acc:%0.3f'%(test_loss,test_acc))

classifier = Clean_dirty_classifier()
classifier.train(3)
classifier.test()




