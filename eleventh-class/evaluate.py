import keras
from keras.models import load_model
from keras.utils import to_categorical
import os
import tifffile
import numpy as np
import random

model = load_model('model.h5')
path = 'D:/ML/adversarial-machine-learning/adversarial-attacks/deepfool/mnist/examples'
deepfool = os.listdir(path)

x_test = np.empty((0,28,28,1))
y_test = np.empty((0))
for file in deepfool:
	image = tifffile.imread(path + '/' + file)
	image = image.reshape(28,28,1)
	x_test = np.append(x_test,[image],axis = 0)
	y_test = np.append(y_test,[10])

y_test = to_categorical(y_test)

predictions = model.predict(x_test)
for i in range(100):
	print(np.argmax(predictions[i]))
test_loss, test_acc = model.evaluate(x_test,y_test)
print(test_loss, test_acc)