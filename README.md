# Crop Disease Prediction Using Machine Learning

This project aims to develop a Convolutional Neural Network (CNN) for predicting crop diseases based on images of crops. The model is trained using TensorFlow and Keras and employs image classification techniques to identify various diseases.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Confusion Matrix](#confusion-matrix)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The objective of this project is to classify images of crops into various categories based on the presence of diseases. The implementation involves using deep learning techniques with TensorFlow and Keras.

## Installation

To run this project, ensure you have Python installed on your system. Then, install the required packages using the following command:

        ```bash
          pip install tensorflow matplotlib seaborn

Usage

Set Up Paths: Update the train_path and valid_path variables in the code to point to your training and validation datasets, respectively.

Display Random Images: Use the display_random_images function to visualize some images from the training and validation datasets.

Build and Train the Model: The CNN model is defined and compiled. The model will be trained using the fit method with early stopping to prevent overfitting.

Evaluate Model: After training, the model is evaluated on the validation dataset, and accuracy metrics are displayed.

Save the Model: The trained model is saved to a file for later use.

Generate Confusion Matrix: Predictions on the validation set are made, and a confusion matrix is generated to visualize the model's performance.

Dataset

The dataset consists of images of crops categorized into subfolders based on different diseases. The images should be organized into training and validation directories. Ensure that the following structure is maintained:


       Crop Disease Dataset/
                ├── train/
                │   ├── Blueberry_Healthy/
                │   │   ├── img1.jpg
                │   │   ├── img2.jpg
                │   ├── Grape_healthy/
                │   └── Potato _Late_blight/
                |   |__ Strawberry_leaf_scorch/
                |   |__ Tomato_Tomato_mosaic_virus/
                └── valid/
                    ├── Blueberry_Healthy/
                    ├── Grape_healthy/
                    └── Potato _Late_blight/
                    |__ Strawberry_leaf_scorch/
                    |__ Tomato_Tomato_mosaic_virus/

Model Architecture

The CNN model is structured as follows:


Conv2D Layer: 32 filters, 3x3 kernel, ReLU activation

MaxPooling2D Layer: 2x2 pool size

Conv2D Layer: 64 filters, 3x3 kernel, ReLU activation

MaxPooling2D Layer: 2x2 pool size

Flatten Layer: Flatten the input for the Dense layer

Dense Layer: 128 units, ReLU activation

Dropout Layer: 50% dropout rate

Output Layer: Dense layer with softmax activation for multi-class classification

Training the Model

The model is trained using the fit method for a maximum of 50 epochs. An EarlyStopping callback is implemented to monitor validation accuracy and stop training when no improvement is observed for 5 consecutive 
epochs.
       
        history = model.fit(
            train_generator, 
            steps_per_epoch=len(train_generator), 
            epochs=50, 
            validation_data=valid_generator, 
            validation_steps=len(valid_generator), 
            callbacks=[early_stopping]
        )

Results

After training, the model's performance is evaluated on the validation dataset. The test loss and accuracy are printed, and training and validation accuracy/loss are plotted.

Confusion Matrix

A confusion matrix is generated to visualize the model's classification performance, allowing for an assessment of true positives, false positives, true negatives, and false negatives.

Contributing

Contributions to this project are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or new features.
