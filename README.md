# Rayan-AI-Course
The Projects of Rayan AI Course.

# HW1: Fraud Detection Analysis (SVM & Logistic Regression)

This project uses an imbalanced dataset with 284,807 transactions, of which only 492 are fraudulent (0.172%), to detect fraud using Support Vector Machines (SVM) and Logistic Regression.

## Dataset Overview
- **Total transactions**: 284,807
- **Fraudulent transactions**: 492 (0.172%)

## Methodology

### 1. Data Loading
Data is shuffled for better separation before analysis.

### 2. Exploratory Data Analysis (EDA)
Key steps include:
- Displaying summary statistics
- Visualizing class imbalance
- Computing a correlation matrix
- Plotting feature distributions

### 3. Preprocessing
Steps to prepare the data include:
- Handling missing values
- Detecting outliers using boxplots
- Normalizing features
- Applying Isolation Forest and IQR for outlier detection

### 4. Model Training
Two models are used for fraud detection:
- **Logistic Regression**: For binary classification
- **Support Vector Machine (SVM)**: Suitable for imbalanced data

### 5. Evaluation
The models are evaluated based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

* * *

- # HW2: Neural Networks and Deep Learning (CNN, Classification, Regression)

## Overview

This repository contains code and demonstrations for Homework 2 of the Machine Learning & Deep Learning course, focusing on Neural Networks and Deep Learning. It includes examples of Binary Classification, Regression, and more complex datasets using PyTorch.

## Contents

1. **Binary Classification**:
   - **Dataset**: Synthetic blobs created with `make_blobs`.
   - **Model**: Simple neural network with one hidden layer.
   - **Training**: Loss function (Binary Cross-Entropy), optimizer (SGD), and training loop.
   - **Evaluation**: Accuracy calculation and decision boundary visualization.

2. **Regression**:
   - **Dataset**: Synthetic regression data generated with `make_regression`.
   - **Model**: A neural network with multiple hidden layers.
   - **Training**: Loss function (Mean Squared Error), optimizer (Adam), and training loop.
   - **Visualization**: Interactive plot of predictions during training.

3. **More Complex Dataset**:
   - **Dataset**: Synthetic S-curve data.
   - **Model**: Modified neural network for complex data.
   - **Training**: Custom training loop to handle complex regression data.

4. **Image Classification**:
   - **Data**: Placeholder for real image data tasks.

- HW3: Variational Autoencoder (Stable Diffusion)
