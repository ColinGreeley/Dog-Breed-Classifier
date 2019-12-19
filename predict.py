from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import sys


path = os.getcwd() + '/' + sys.argv[1]
data_path = '~/ML_Demo/data/test'

# extract convolutional base from Google's InceptionV3 object detection model
cb = InceptionV3(weights='imagenet', 
                  include_top=False, 
                  input_shape=(299, 299, 3))
last_layer = layers.GlobalAveragePooling2D()(cb.output)
conv_base = Model(cb.input, last_layer)


# loading the saved model from memory
model1 = tf.keras.models.load_model('~/ML_Demo/DogBreedModel2.h5')

# create data generator
raw_image = Image.open(path)
raw_image = raw_image.rotate(270)
resized_image = raw_image.resize((299, 299))
image = np.expand_dims(np.asarray(resized_image).astype('float32'), axis=0) / 255.0

features = conv_base.predict(image)
features = np.reshape(features, (1, 2048))
prediction = model1.predict(features)
class_names = os.listdir(data_path)

print(class_names[np.argmax(np.array(prediction))])
plt.figure()
plt.imshow(raw_image)
plt.xlabel(class_names[np.argmax(np.array(prediction))])
plt.show()
