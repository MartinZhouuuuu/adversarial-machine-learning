import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model,load_model
from keras.preprocessing.image import img_to_array
import tifffile
import os
import random


image = tifffile.imread(
	'/Users/apple/Documents/adversarial-machine-learning/full-fgsm/test/original/5121.tif'
	)
model = load_model('/Users/apple/Desktop/8350/D.h5')
prediction = model.predict(image.reshape(1,28,28,1))
print(prediction)