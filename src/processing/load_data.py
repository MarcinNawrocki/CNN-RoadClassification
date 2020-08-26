import os
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image

def get_data(image_shape = (220,220,3), batch_size = 16):
    """Loads a dataset for the road classification problem using Keras ImageDataGenerator

    Keyword Arguments:
        image_shape {tuple} -- shape to which images should be resized (default: {(220,220,3)})
        batch_size {int} -- batch size for learning (must be specified now, because of using Keras generator) (default: {16})

    Returns:
        [DirectoryIterator] -- train dataset ready to pass into Keras fit_generator function
        [DirectoryIterator] -- validation dataset ready to pass into Keras fit_generator function
    """

    path_data = "./data/processed"
    path_train_data = os.path.join(path_data,'train')
    path_val_data = os.path.join(path_data,'val')

    #define generator with data augmentation for training dataset
    train_image_gen = ImageDataGenerator(rescale=1. /255,# Rescale the image by normalzing it
                                     horizontal_flip = True,
                                     vertical_flip= True,
                                     brightness_range=[0.2,1.0],
                                     featurewise_center=True,
                                     featurewise_std_normalization=True
                              )
               
    #define generator for validation dataset
    val_image_gen = ImageDataGenerator(rescale=1. /255, # Rescale the image by normalzing it
                              )
    #fit train_image_gen to use featurewise_center and featurewise_std_normalization
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(path_train_data):
        file_list += [os.path.join(dirpath, file) for file in filenames]
    X = []
    for file in file_list:
        tmp = load_img(file,target_size = image_shape)
        np_tmp = img_to_array(tmp)
        X.append(np_tmp)
    X_train = np.array(X)
    X_train.shape

    train_image_gen.fit(X_train)

    #load datasets using generators
    train_images = train_image_gen.flow_from_directory(path_train_data,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=batch_size)

    val_images = val_image_gen.flow_from_directory(path_val_data,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=batch_size,
                                                shuffle = False)

    return train_images, val_images


#use method from Keras instead of PIL
def get_sample(image_shape):
    """Loads a randomly choosed sample of road image

    Arguments:
        image_shape {tuple} -- shape to which images should be resized

    Returns:
        [NumPy array] -- randomly choosed sample image
    """

    NUMBER_OF_SAMPLES = 38
    SAMPLE_PATH = "./data/samples/"
    EXTENSION = ".PNG"
    sample_number = random.randint(0,NUMBER_OF_SAMPLES)
    sample_file_name = str(sample_number) + EXTENSION
    sample = load_img(os.path.join(SAMPLE_PATH,sample_file_name))
    sample = sample.resize(image_shape[:2])
    np_sample = img_to_array(sample)/255 
    #np_sample = np.expand_dims(np_sample,axis=0)
    
    return np_sample
