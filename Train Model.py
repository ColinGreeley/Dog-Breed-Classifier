from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
import numpy as np
import tensorflow as tf


path = '~/ML_Demo/newdata/'

train_dir = path + 'train/'
test_dir = path + 'test/'
val_dir = path + 'val/'

# extract convolutional base from Google's InceptionV3 object detection model
cb = InceptionV3(weights='imagenet', 
                  include_top=False, 
                  input_shape=(299, 299, 3))
last_layer = layers.GlobalAveragePooling2D()(cb.output)
conv_base = Model(cb.input, last_layer)
#conv_base.summary()

# creating trainable model to replace the head of InceptionV3
model = models.Sequential()
#model.add(layers.Dense(1024, activation='relu', input_dim = 2048))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(120, activation='softmax', input_dim = 2048))

model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# create data generator
datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
batch_size = 32

# returns the output of the last layer of the InceptionV3 convolutional neural network
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 2048), dtype=np.float32)
    labels = np.zeros(shape=(sample_count), dtype=np.float32)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='sparse')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# InceptionV3 output tensor for 10316 inputs
train_features, train_labels = extract_features(train_dir, 17160)
#validation_features, validation_labels = extract_features(val_dir, 6844)
#test_features, test_labels = extract_features(test_dir, 3420)

train_features = np.reshape(train_features, (17160, 2048))
#validation_features = np.reshape(validation_features, (6844, 2048))
#test_features = np.reshape(test_features, (3420, 8 * 8 * 2048))



# fit the new model we have made to the training and validation data
for i in range(6):
    model.fit(train_features, train_labels, 
                epochs=5,
                batch_size=64)
            #validation_data=(validation_features, validation_labels))

#model.evaluate(test_features, test_labels, verbose=2)
    model.save('DogBreedModel2.{}.h5'.format(i))
