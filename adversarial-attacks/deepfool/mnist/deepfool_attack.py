from __future__ import print_function
global sess
global graph
import tensorflow as tf
import numpy as np
import keras
from cleverhans.attacks import DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import History
from keras import backend as K

img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

csvFile1, csvFile2 = [], []
with tf.Graph().as_default():
  with tf.Session() as sess:
    K.set_session(sess)
    model = keras.models.load_model('model.h5')

    # Build dataset to be attacked
    adv_inputs, adv_labels = np.empty(x_test.shape), np.empty(y_test.shape)
    j = 0
    for i in range(0,x_test.shape[0]):
      if np.argmax(model.predict(x_test[i:i+1])) == np.argmax(y_test[i]):
        adv_inputs[j] = x_test[i]
        adv_labels[j] = y_test[i]
        # csvFile1.append([[i,j]])
        j += 1
    adv_inputs = adv_inputs[:100]
    adv_labels = adv_labels[:100]
    print("Legitimate test accuracy = %0.3f" % (j/y_test.shape[0]))
    print("Dataset of %d to be attacked." % adv_inputs.shape[0])
    print(adv_inputs.shape, adv_labels.shape)  

    # Attack
    wrap = KerasModelWrapper(model)
    deepfool = DeepFool(wrap, sess=sess)
    params = {}
    x_adv_1 = deepfool.generate_np(adv_inputs[:20], **params)
    x_adv_2 = deepfool.generate_np(adv_inputs[20:40], **params)
    x_adv_3 = deepfool.generate_np(adv_inputs[40:60], **params)
    x_adv_4 = deepfool.generate_np(adv_inputs[60:80], **params)
    x_adv_5 = deepfool.generate_np(adv_inputs[80:], **params)
    x_adv = np.concatenate((x_adv_1, x_adv_2, x_adv_3, x_adv_4, x_adv_5), axis=0)
    score = model.evaluate(x_adv, adv_labels, verbose=0)
    print('Adv. Test accuracy: %0.3f' % score[1])

    # Initialize random choosing of adversarial images
    num_examples = 100

    index_list = list(range(x_adv.shape[0]))
    import random
    random.seed(9123)
    random.shuffle(index_list)

    # Save adversarial images
    import tifffile

    classes = {}
    for i in range(0, 10):
      classes[i] = 0

    j = 0
    print("\nSaving %d adversarial examples -------------" % num_examples)
    for i in index_list:
      if np.argmax(model.predict(x_adv[i:i+1])) != np.argmax(adv_labels[i]) and classes[np.argmax(adv_labels[i])] < 105:
        temp_array1 = x_adv[i].astype(np.float32)
        tifffile.imsave("examples/%d.tif" % j, temp_array1)
        temp_array = adv_inputs[i].astype(np.float32)
        tifffile.imsave("originals/%d.tif" % j, temp_array)
        # csvFile2.append([[i,j]])
        j += 1
        classes[np.argmax(adv_labels[i])] = classes[np.argmax(adv_labels[i])] + 1
      if j == num_examples:
        print("Generated %d adversarial images." % num_examples)
        break

    # Print class distrbution
    print("Class distribution:")
    for i in range(0, 10):
      print("%d, %d" % (i, classes[i]))
