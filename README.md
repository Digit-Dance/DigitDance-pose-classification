# ECK2024: A Serious Game for Dementia Prevention

## Overview
This project is part of a larger initiative to develop **A Serious Game for Dementia Prevention** that combines physical and cognitive exercises. The game incorporates pose classification as a key feature to identify and evaluate physical movements performed by users while solving arithmetic tasks. This README provides an overview of the pose classification component, including its workflow, data preprocessing, and model training.

1. [Project Objective](#project-objective)
2. [Workflow](#workflow)
   - [1. Data Preprocessing](#1-data-preprocessing)
     - [1.1 Labeling](#11-labeling)
     - [1.2 Data Augmentation](#12-data-augmentation)
     - [1.3 Skeletonization](#13-skeletonization)
   - [2. Model Training](#2-model-training)
     - [2.1 Dataset Preparation](#21-dataset-preparation)
     - [2.2 CNN Model Definition](#22-cnn-model-definition)
     - [2.3 Training and Evaluation](#23-training-and-evaluation)
3. [How to Run](#how-to-run)

## Project Objective
The primary goal of the pose classification model is to:
- Accurately classify **10 distinct poses** captured during gameplay.
- Integrate seamlessly with the **Cognicise game** to evaluate user actions in real-time.
- Enhance the effectiveness of cognitive and physical training.
  

## Workflow

### **1. Data Preprocessing**
#### 1.1 Labeling
- Organized raw images into class folders (`0-9`), each representing a unique pose.
- Created `labels.csv` to map image file paths to corresponding pose labels.

#### 1.2 Data Augmentation
- Performed data augmentation using **Albumentations** to enhance model robustness:
  - Random rotation (±30°).
  - Scaling (±20%).
  - Salt-and-pepper noise.

#### 1.3 Skeletonization
- Used **Mediapipe** to extract human pose landmarks from augmented images.
- Selected specific landmarks to focus on critical features for pose classification.
- Connected key landmarks with colored lines to create a skeletal representation.


### **2. Model Training**
#### 2.1 Dataset Preparation
- Uploaded skeletonized images and `skeletonized_labels.csv` to Google Colab.
- Normalized and resized images to 28x28 pixels for consistency.
- Split the data into training and testing sets with an 8:2 ratio.

#### 2.2 CNN Model Definition
The CNN model is built using **Flax**, a neural network library powered by **JAX**, with training optimized through **Optax**. The model structure includes:

- **Convolutional Layers**:
  - 3 layers for feature extraction with ReLU activation functions.
  - Average pooling to reduce dimensions after each layer.
- **Fully Connected Layers**:
  - 1 layer with 256 neurons.
  - Output layer with 10 neurons, representing each pose category.
- **Optimizer**:
  - Adam optimizer from Optax with a learning rate of 0.001.

#### 2.3 Training and Evaluation
- **Loss Function**: Softmax cross-entropy loss using Optax.
- **Training**: Trained over 20 epochs with real-time computation of loss and accuracy for each batch.
- **Evaluation**: Assessed the model on the test dataset, measuring final loss and accuracy to ensure robust performance.


## How to Run

### **1. Clone the Repository**
```
git clone <repository-url>
cd <repository-folder>
```

### **2. Set Up Environment**
Install the required Python libraries using requirements.txt:
```
pip install -r requirements.txt
```

### **3. Preprocess Data**
Run the following scripts in order to preprocess the data:

1. `labeling.py`: Generates `labels.csv` from the raw dataset.
2. `augmentation.py`: Performs data augmentation on the labeled images.
3. `skeletonization.py`: Creates skeletonized images and generates `skeletonized_labels.csv`.

### **4. Upload Processed Data**
Upload the `skeletonized_labels.csv` and `skeletonized_image folder` to **Google Colab**.

### **5. Train the Model**
Run the training script: `python train_model.ipynb`         
If you're using Google Colaboratory (Colab), enable the GPU acceleration (**Runtime > Change runtime type > Hardware accelerator:GPU**).
