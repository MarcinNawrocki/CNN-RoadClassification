# CNN_RoadClassification

## General info
The goal of the project was to develop a tool which will be classified the road types based on their images, using Convolutional Neural Networks. Roads were splitted 
into four classes:
1. asphalt road,
2. cubblestones road,
3. paved road,
4. unpaved road.

## Technologies
* tensorflow-gpu 2.1.0;
* Keras-Applications 1.0.8
* Keras-Preprocessing==1.1.0
* Pillow 7.1.2;
* scikit-learn 0.23.0;
* numpy 1.18.5;

## Database
Due to pandemy, only limited database was collected via Google Street View API (original plan was to gathered inreal roads photos):
* class 1 (asphalt road) 210 photos,
* class 2 (cubblestones road) 204 photos,
* class 3 (paved road) 200 photos,
* class 4 (unpaved road) 200 photos.

Database for class 3 and 4 contain many photos of the same roads in different places and angles, which was significant issue during learning.
Database is available on [Google Drive.](https://drive.google.com/drive/folders/13rW96ipZKy81_RhFfgUYJaLXNAARp4Kh?usp=sharing) 

## CNN
Architecture of CNN, generated by [Netron](https://github.com/lutzroeder/netron) is show below:

![alt text](https://github.com/MarcinNawrocki/CNN_RoadClassification/blob/master/Model_scheme.png "CNN architecture")

The file which contained CNN model is available on [Google Drive](https://drive.google.com/file/d/1uiMJwL5XJDJuF70HBgoO81flYOa7GQEO/view?usp=sharing) as Keras ".h5" file.
Apart from architecture, some other functioalities from Keras was used: 
* L2 regularization on dense layers with rate = 0.001.
* Data Augmentation.
* Changing initialzator in convolutional layers to "Kaiming initializer" which is called "he_uniform" in Keras API (based on [this article](https://towardsdatascience.com/why-default-cnn-are-broken-in-keras-and-how-to-fix-them-ce295e5e5f2)).
* Early Stopping feature with patience equal to 2 epochs.

## Fine-tunning model
Other implementations was fine-tunning based aproach on VGG-16 network, which was previosly learned on ImageNet database. All convolutional layers was frozen and learning was
perfromed only on 2 Dense layers (512 and 4 neurons, with 0.5 dropout) for 50 epochs.

## Results
CNN created from scratch achieved 84,8% accuracy on test dataset, while fine-tunning aproach reached 90,5%.

## Repository content
Repository contains a number of jupyter notebooks sheets:
* Predict.ipynb is sheet prepare to test our model.
* database preprocess folder contains some sheets, which was used to organize database to form accepted by flow_from_directory Keras method.
* Training.ipynb was used to perform learn and evaluate different architectures
* Training_FineTunning.ipynb served for perform fine-tunning on VGG-16 model.
