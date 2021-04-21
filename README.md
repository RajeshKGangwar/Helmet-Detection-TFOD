# Helemt Detection with Tensorflow Object Detection:
![Python 3.6](https://img.shields.io/badge/Python-v3.6-green) ![Opencv-Python](https://img.shields.io/badge/OpenCv--Python-v4.5-red) ![Tensorflow](https://img.shields.io/badge/tensorflow-1.14-brightgreen) ![Keras](https://img.shields.io/badge/Keras-1.0-yellowgreen)

## Table of Content
  * [Application Demo](#Application-demo)
  * [Overview](#overview)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Credits](#credits)


## Demo
![output4](https://user-images.githubusercontent.com/63975688/115552496-07082580-a2ca-11eb-928b-cbc377d0839e.jpg)


## Overview
This is a Simple App for detecting whether a person is wearing helmet or not. The App has been Designed using Python and flask Api. The model is created by Transfer learning approach, a pretrained model as __sssd_mobilenet_v1__ is used for model building. <br>
The App Captures live frame from webcam/device camera and returns the value of bounding boxes with confidence score if the person is wearing the helmet.


## Technical Aspect
This App is divided into two part:
1. __app.py__ : Entry point of the App with all the initials.


2. __Prediction.py__: This file gets frames per second as input and processes them to detect helmet as final result.
    

## Installation
The Helmet Detection App is coded in python version 3.6, with other libraries as keras, tensorflow, numpy etc. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). Upgrade using pip package if you are using any lower version. The dependencies are mentioned in the requirements.txt file. Go with the below mentioned command for installing them in one go.
```bash
pip install -r requirements.txt
```

## Bug / Feature Request

If you find some bug/defect or if you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/RajeshKGangwar/Helmet-Detection-TFOD/issues).

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

<p align="left"> <a href="https://www.w3schools.com/css/" target="_blank"></a> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-ar21.svg" alt="open-cv" width="150" height="150"/> <img src="https://www.vectorlogo.zone/logos/numpy/numpy-ar21.svg" alt="numpy" width="150" height="150"/>
</a> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-ar21.svg" alt="tensorflow" width="150" height="150"/>


## Credits

- [Tensorflow Object detection Model](https://github.com/tensorflow/models) 
- [Sample Dataset From Google](https://google.com)
