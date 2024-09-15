# Rayan-AI-Course
The Projects of Rayan AI Course.

# HW1: Fraud Detection Analysis (SVM & Logistic Regression)

This project involves analyzing a highly imbalanced dataset to detect fraudulent transactions using Support Vector Machines (SVM) and Logistic Regression. The dataset includes transactions over a two-day period, consisting of 284,807 records, of which 492 are fraud cases. Fraudulent transactions make up only 0.172% of the total, presenting a significant class imbalance challenge.

## Dataset Overview
The dataset used for this project has the following characteristics:
- **Total transactions**: 284,807
- **Fraudulent transactions**: 492
- **Fraud rate**: 0.172% (highly imbalanced dataset)

## Steps and Methodology

### 1. Data Loading
We begin by loading the dataset and shuffling the data to ensure better separation for training and testing phases.

### 2. Exploratory Data Analysis (EDA)
The Exploratory Data Analysis helps in understanding the dataset’s structure and characteristics before proceeding with model training.

- **Summary statistics**: Display key statistical metrics such as mean, standard deviation, and distribution for each feature.
- **Class distribution visualization**: Visualize the imbalance in class distribution between fraudulent and non-fraudulent transactions.
- **Class distribution percentages**: Provide a breakdown of the class distribution as percentages.
- **Correlation matrix**: Compute and visualize the correlation matrix to identify relationships between features.
- **Feature distributions**: Plot the distributions of key features to understand the data's behavior.

### 3. Preprocessing
Before training the models, several preprocessing steps are performed to ensure the dataset is clean and ready for analysis:

- **Handling missing values**: Identify any null columns and handle them accordingly.
- **Outlier detection**: Use boxplots to detect and visualize outliers.
- **Normalization**: Normalize feature values to ensure consistency across the dataset.
- **Isolation Forest**: Apply the Isolation Forest algorithm for anomaly detection and outlier identification.
- **IQR technique**: Use the Interquartile Range (IQR) technique to further refine outlier detection.

### 4. Model Training
Two machine learning models are implemented to detect fraud:

- **Logistic Regression**: A simple yet powerful model suitable for binary classification problems like fraud detection.
- **Support Vector Machine (SVM)**: A robust classifier that works well with high-dimensional spaces and is particularly useful for imbalanced datasets.

Each model is trained and evaluated for its performance in detecting fraud.

### 5. Evaluation
The models are evaluated based on appropriate metrics for imbalanced datasets, such as:

- **Accuracy**: Overall correctness of the model.
- **Precision**: The percentage of correctly identified frauds out of all predicted frauds.
- **Recall**: The percentage of actual frauds that were correctly identified by the model.
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced evaluation metric.

- HW2: Neural Networks and Deep Learning (CNN, Classification, Regression)
- HW3: Variational Autoencoder (Stable Diffusion)
