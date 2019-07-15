import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model,load_model
from keras.preprocessing.image import img_to_array
import tifffile
import os
import random

image = tifffile.imread(
	'/Users/apple/Google Drive/adversarial-machine-learning/patches/test/original/49.tif'
	)
model = load_model('model-files/classifier-5.h5')
prediction = model.predict(image.reshape(1,5,5,1))
print(prediction)