# Mask-Detection-System
A Convolutional Neural Network trained to detect a mask, using a Flask API.

## Requirements
Flask

Tensorflow 2.0.0

Keras

## Overview
A Flask API is created, and a model is trained with a dataset containing augmented images of people with and without masks.
A convolutional neural network is trained for the purpose of detecting the output accurately.
The model weights and model architecture are downloaded to create an object of the model.
An instance of the object is called to detect a mask, when the device camera is opened.

### Dataset
The [dataset](https://github.com/prajnasb/observations/tree/master/experiements/data) contained augmented images of individuals, with masks added, and without masks.

