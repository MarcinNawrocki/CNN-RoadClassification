import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report,confusion_matrix

def evaluate_model (model, test_images, show_conf_matrix=False):
    """Perform evaluation of a given model using test_images dataset

    Arguments:
        model {[Sequential]} -- trained keras model to evaluate
        test_images {[DirectoryIterator]} -- dataset to evaluate a model performance

    Keyword Arguments:
        show_conf_matrix {bool} -- if true confusion matrix is additionaly shown in an elegant way using seaborn.heatmap() function (default: {False})

    Returns:
        [NumPy array] -- Confusion Matrix of a given model
        [str] -- Classification Report of a given model from sklearn library
    """
    model.evaluate_generator(test_images)
    predicted_class_indices = get_predictions_indices(model,test_images)
    conf_matrix = confusion_matrix(test_images.classes, predicted_class_indices)
    if show_conf_matrix:
        plt.figure(figsize=(10,6))
        sns.heatmap(conf_matrix, annot=True)
    classification_rep = classification_report(test_images.classes, predicted_class_indices)

    return conf_matrix, classification_rep

def get_predictions_indices(model,test_images):
    """[]

    Arguments:
        model {[Sequential]} -- trained keras model to perform prediction
        test_images {[DirectoryIterator]} -- dataset to predict using given model

    Returns:
        [{NumPy array}] -- predicted class indice for each sample in a test_images dataset
    """
    pred= model.predict_generator(test_images)
    predicted_class_indices=np.argmax(pred,axis=1)

    return predicted_class_indices

def save_model(model, path):
    """Saving Keras model in a specified path as ".h5" file

    Arguments:
        model {[Sequential]} -- keras model to save
        path {[str]} -- path including a filename
    """

    #TODO check if path ends with .h5
    model.save(path)

def predict (model, np_sample):
    """Making prediction on a single image using given model

    Arguments:
        model {[Sequential]} -- trained, keras model to perform prediction
        np_sample {[NumPy array]} -- Four dimensional image as NumPy array

    Returns:
        [str] -- predicted class name
    """
    labels = {0:"Asphalt", 1:"Cubblestones", 2:"Paved", 3:"Unpaved"}
    #model prediction return as one-hot encoding
    np_sample_4d = np.expand_dims(np_sample, axis=0)
    one_hot_prediction = model.predict(np_sample_4d)
    #class prediction number
    prediction = np.argmax(one_hot_prediction,axis=1)
    predicted_class = labels[prediction[0]]
    return predicted_class