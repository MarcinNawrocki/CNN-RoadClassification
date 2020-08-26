import os

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping   
from tensorflow.keras.models import load_model



def train_model(train_images, val_images, image_shape, conv_initializer, dense_reg_rate):
    """Load and trains a model for the road classification problem. Function execution could take a signifficant amount of time,
        due to performing neural netowork learning (exact time depends on your hardware)

    Arguments:
        train_images
        val_images
        input_shape {[tuple]} -- input shape of network, should be equal to image_shape from dataset
        conv_initializer {[keras initializer]} -- weights initializer for convolutional layers
        dense_reg_rate {[float]} -- L1 regularization rate for dense layers

    Returns:
        [Sequential] -- already trained keras model for the road classification problem
    """
    model = get_model(image_shape, conv_initializer, dense_reg_rate)

    early_stop = EarlyStopping(monitor='val_loss',patience=2, restore_best_weights=True)
    results = model.fit_generator(train_images,epochs=50,
                                  validation_data=val_images,
                                 callbacks=[early_stop])
    return model

def get_model (input_shape, conv_initializer, dense_reg_rate):
    """Load compiled model, for road classfication problem

    Arguments:
        input_shape {[tuple]} -- input shape of network, should be equal to image_shape from dataset
        conv_initializer {[keras initializer]} -- weights initializer for convolutional layers
        dense_reg_rate {[float]} -- L1 regularization rate for dense layers

    Returns:
        [Sequential] -- compiled, keras model  for the road classification problem
    """
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5,5),input_shape=input_shape, activation='relu', strides=(2,2), 
                    kernel_initializer=conv_initializer))

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
                    kernel_initializer=conv_initializer))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', 
                    kernel_initializer=conv_initializer))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', 
                    kernel_initializer=conv_initializer))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                    kernel_initializer=conv_initializer))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                    kernel_initializer=conv_initializer))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                    kernel_initializer=conv_initializer))
    model.add(MaxPooling2D(pool_size=(3, 3)))
                        
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                    kernel_initializer=conv_initializer))
            
    model.add(Flatten())

    model.add(Dense(384, activation='relu', kernel_regularizer=l2(dense_reg_rate)))
    #Final layer
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model

def load_trained_model():
    """Load previously trained model, for the road classification problem from ".h5" file

    Returns:
        Sequential -- trained, keras model for the road classification problem
    """
    #check later with notebook, because it is possible to have differences with cwd
    model = load_model("./model/road_classification_model.h5")
    return model