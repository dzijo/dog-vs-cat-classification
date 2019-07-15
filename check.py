from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

def predict(image):
    # load the image
    img = load_image(image)
    # load model
    model = load_model('final_model_TL.h5')
    # predict the class
    result = model.predict(img)
    return result[0][0].astype(int)

print(predict('66832444_2275116232582234_7067699075927244800_n.jpg'))


#model = load_model('final_model_TL.h5')
#
#test_datagen = ImageDataGenerator(rescale=1.0/255.0)
#
#test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
#    class_mode='binary', batch_size=32, target_size=(224, 224))
#
#a, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
#print('> %.3f' % (acc * 100.0))
#print(a)
