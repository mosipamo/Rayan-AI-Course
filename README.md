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

# HW2: Neural Networks and Deep Learning (CNN, Classification, Regression)

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
### Image Classification Overview
We’ll be working with a cleaned version of the Cats vs. Dogs dataset, organized into train and test folders with subfolders for CAT and DOG images. The dataset files are prepared for use in training and evaluating image classification models.

### Key Steps in Image Processing
1. **Examine the Data**: Use Python’s `os` module and Pillow to gather image data and sizes, then summarize using a pandas DataFrame.
2. **Preprocess Images**:
   - **Aspect Ratio & Scaling**: Adjust images to fit desired dimensions, e.g., 224x224 pixels.
   - **Normalization**: Convert pixel values to a [0,1] range and normalize using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
   - **Transformations**: Apply operations like resizing, cropping, flipping, and rotation to augment data.

### Image Data Preparation
1. **Create a DataFrame**: Extract and analyze image sizes to help choose model parameters.
2. **Define and Apply Transformations**: Use PyTorch's `transforms` for tensor conversion, resizing, cropping, and normalization.

### Convolutional Neural Network (CNN)
1. **Define the Model**: Design a CNN architecture using layers such as `nn.Conv2d`, `nn.Linear`, `F.max_pool2d`, and `F.relu`.
2. **Training**:
   - **Instantiate the Model**: Create a CNN model, set up loss and optimization functions.
   - **Train the Model**: Limit training batches and epochs for efficiency.
3. **Evaluate**: Test the model's performance on validation data.

### Using Pretrained Models
1. **Download and Modify Pretrained Models**: Utilize models like AlexNet, VGG, ResNet from `torchvision.models`. Freeze feature parameters and adjust the classifier to output two categories (CAT and DOG).
2. **Train the Modified Model**: Focus on training the classifier while keeping the feature extraction layers frozen.

### Final Steps
1. **Run New Images Through the Model**: Test the model with new images to validate performance and predictions.

* * *

- HW3: Variational Autoencoder (Stable Diffusion)
