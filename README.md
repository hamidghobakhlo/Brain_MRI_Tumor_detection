# Brain MRI Tumor Detection

This repository contains the code and resources for detecting brain tumors from MRI images using deep learning techniques. The project aims to provide an efficient and accurate solution for assisting medical professionals in diagnosing brain tumors, complete with an interactive Streamlit web application.


---

## Table of Contents
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Model Architecture](#model-architecture)  
5. [Limitations and Future Work](#limitations-and-future-work)
6. [Streamlit Web Application](#streamlit-web-application)  
7. [Connect with Me](#connect-with-me)

---

## Introduction

Brain tumors are among the most severe health challenges, requiring precise and timely detection to increase treatment success rates. This project applies advanced image processing and machine learning techniques to automate the identification of tumors in MRI scans. The model is designed to enhance diagnostic efficiency and accuracy using a custom Convolutional Neural Network (CNN) trained on MRI datasets.

---

## Features

- **Robust Image Preprocessing**: Automatic operations include resizing MRI scans to `(128, 128, 3)` (RGB) and converting labels using `OneHotEncoder`.  
- **Deep Learning Integration**: Utilizes a custom Sequential CNN featuring `Conv2D`, `BatchNormalization`, `MaxPooling2D`, and `Dropout` layers with the `Adamax` optimizer and `categorical_crossentropy` loss.  
- **Model Persistence**: Supports saving and loading trained models (`brain_tumor_model.h5`) for seamless inference.  
- **User-Friendly Web App**: Includes an interactive Streamlit application that allows users to upload MRI images and instantly view prediction results and confidence scores.  

---

## Dataset

The dataset is structured and categorized into two classes:  
1. **Tumorous Images (`yes`)**  
2. **Non-Tumorous Images (`no`)**  

**Details**:  
-
- **Labels / Classes**: `0 - Tumor`, `1 - Normal`
- **Data Shape & Splitting**: Total dataset reshaped to `(2891, 2)`, split via `train_test_split` with `test_size=0.2`, `shuffle=True`, and `random_state=42`.
- **Preprocessing Workflow**:  
  - Resizing all input images to `128x128x3` pixels for uniformity.  
  - Encoding categorical labels using `OneHotEncoder`.  

---

## Model Architecture

The neural network is designed as a Sequential model with the following architecture:  

1. **Input Layer**: Processes RGB MRI scans resized to `128x128x3` pixels.  
2. **Convolutional & Normalization Layers**: Extracts critical spatial features using `Conv2D` coupled with `BatchNormalization`.  
3. **Pooling & Dropout Layers**: Uses `MaxPooling2D` to reduce spatial dimensions and `Dropout` layers to mitigate overfitting risks.  
4. **Flatten & Dense Layers**: Integrates extracted features through fully connected layers for final decision-making.  
5. **Output Layer**: Uses `softmax` (via categorical cross-entropy) to classify images into categories (`Tumor` or `No Tumor`) optimized with the `Adamax` optimizer.

---
## Limitations and Future Work

Given the data-driven nature of deep learning models, the final accuracy of the system is directly tied to the diversity and volume of the dataset. In the current phase of this research, due to limited access to a large-scale volume of MRI data, the system has been developed as an initial prototype. To improve accuracy levels and increase model reliability for clinical applications, future development strategies include:

- **Dataset Expansion**: Collaborating with medical centers and imaging clinics to acquire and label diverse, real-world MRI images in order to enhance the model's generalization capability.
- **Continuous Optimization**: Retraining the model using newly acquired data and employing advanced data augmentation techniques to address potential class imbalances.
- **Clinical Adaptation**: Refining the network architecture based on feedback from radiology specialists and evaluating model performance in real clinical environments.
---

## Streamlit Web Application

An interactive web application built with **Streamlit** is included for real-time inference. 

- **How to Run**:
  1. Ensure dependencies (`streamlit`, `tensorflow`, `numpy`, `PIL`) are installed.
  2. Run the app using the command:
```bash
streamlit run app.py
```
## Connect with Me

**Hamid Ghobakhlo**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hamidghobakhlo/)
