import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import tifffile
import os
import random
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 11
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32').reshape(60000,28,28,1)
x_test = x_test.astype('float32').reshape(10000,28,28,1)

#load the 11th class
generated_path = 'D:/ML/adversarial-machine-learning/generated-images'

generated_train = os.listdir(generated_path + '/train')
train_picks = random.choices(generated_train, k = 6000)
for file in train_picks:
	image = tifffile.imread(generated_path + '/train/' + file)
	image = image.reshape(28,28,1)
	x_train = np.append(x_train,[image],axis = 0)
	y_train = np.append(y_train,[10])

generated_test = os.listdir(generated_path + '/test')
test_picks = random.choices(generated_test, k = 1000)

for file in test_picks:
	image = tifffile.imread(generated_path + '/test/' + file)
	image = image.reshape(28,28,1)
	x_test = np.append(x_test,[image],axis = 0)
	y_test = np.append(y_test,[10])

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('y_train shape', y_train.shape)

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
				 activation='relu',
				 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
			  optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])

history = model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(x_test, y_test),
		  shuffle = True
		  )

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', round(score[0],3))
print('Test accuracy:', round(score[1],3))
model.save('model.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Model's Training & Validation accuracy across epochs")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('accuracy.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model's Training & Validation loss across epochs")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc = 'upper right')
plt.savefig('loss.png')
plt.close()
