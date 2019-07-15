'''this is a trial on MNIST dataset
which is to directly classify clean and dirty images
unlikely to succeed'''
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD, Adam, rmsprop
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import tifffile
import os
import random
class Clean_dirty_classifier():
	def __init__(self,rows,columns):
		self.num_of_rows = rows
		self.num_of_columns = columns
		self.image_shape = (self.num_of_rows,self.num_of_columns,1)
		self.optimizer = Adam()
		self.num_of_iterations = 100
		self.classifier = self.classifier()
		self.train_loss_array = np.empty((0,1))
		self.train_acc_array = np.empty((0,1))
		self.val_loss_array = np.empty((0,1))
		self.val_acc_array = np.empty((0,1))

	def get_dataset(self,path,sample_size):
		dataset = np.empty((0,self.num_of_rows,self.num_of_columns,1))
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
		best_val_acc = 0
		for epoch in range(epochs):
			epoch_loss = 0
			epoch_acc = 0
			for iteration in range(self.num_of_iterations):
				train_x, train_y = self.get_dataset(
					'patches/train',
					128
					)
				train_return = self.classifier.train_on_batch(train_x,train_y)
				epoch_loss += train_return[0]
				epoch_acc += train_return[1]


			epoch_loss /= self.num_of_iterations
			epoch_acc /= self.num_of_iterations
		
			print('Epoch%d train_loss:%0.3f train_acc:%0.3f'%(epoch+1, epoch_loss,epoch_acc))
			self.train_loss_array = np.append(self.train_loss_array,np.array([[epoch_loss]]),axis=0)
			self.train_acc_array = np.append(self.train_acc_array,np.array([[epoch_acc]]),axis=0)
		
			#validation
			val_x, val_y = self.get_dataset(
				'patches/test',
				500
				)

			val_loss,val_acc = self.classifier.evaluate(val_x,val_y)
			self.val_loss_array = np.append(self.val_loss_array,np.array([[val_loss]]),axis=0)
			self.val_acc_array = np.append(self.val_acc_array,np.array([[val_acc]]),axis=0)
			print('val_loss:%0.3f val_acc:%0.3f'%(val_loss,val_acc))
			#model checkpointer
			if val_acc >= best_val_acc:
				print('val acc increased from %0.3f to %0.3f'%(best_val_acc,val_acc))
				best_val_acc = val_acc
				self.save_model()
				print('model saved')
			else:
				print('val acc did not increase')
			self.plot_losses()

	def save_model(self):
		self.classifier.save('model-files/classifier-5.h5')


	def test(self):
		test_x, test_y = self.get_dataset(
					'patches/test',5000)
		best_model = load_model('model-files/classifier-5.h5')
		test_loss,test_acc = best_model.evaluate(test_x,test_y)
		print('test_loss:%0.3f test_acc:%0.3f'%(test_loss,test_acc))


	def plot_losses(self):
		plt.plot(self.train_acc_array[:,0])
		plt.plot(self.val_acc_array[:,0])
		plt.title("Accuracy")
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc = 'upper left')
		plt.savefig("plots/accuracy.png")
		plt.close()

		plt.plot(self.train_loss_array[:,0])
		plt.plot(self.val_loss_array[:,0])
		plt.title("Loss")
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc = 'upper right')
		plt.savefig("plots/loss.png")
		plt.close()
		print('plots saved')

	def sample_scores(self):
		row, column = 2,10
		test_x, test_y = self.get_dataset(
					'patches/test',column)
		best_model = load_model('model-files/classifier-5.h5')
		
		batch_adv = test_x[:column]
		validity_adv = best_model.predict(batch_adv)
		batch_real = test_x[column:]
		validity_real = best_model.predict(batch_real)
		fig, axs = plt.subplots(row,column)
		for j in range(column):
			axs[0,j].imshow(batch_real[j,:,:,0],cmap = 'gray')
			axs[0,j].axis('off')
			axs[0,j].set_title('%0.3f'%validity_real[j,1],size = 12)
			
			axs[1,j].imshow(batch_adv[j,:,:,0],cmap = 'gray')
			axs[1,j].axis('off')
			axs[1,j].set_title('%0.3f'%validity_adv[j,1],size = 12)

		fig.text(0.5, 0.6, 'real',fontsize = 15)
		fig.text(0.45, 0.1, 'adversarial',fontsize = 15)
		fig.savefig('sample.png')
		plt.close()
classifier = Clean_dirty_classifier(5,5)
# classifier.train(20)
# classifier.test()
classifier.sample_scores()



