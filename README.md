# Gesture Recognition for Smart TVs

## Project Overview

The goal of this project is to develop a gesture recognition feature for smart TVs, allowing users to control the TV without a remote using five specific gestures. Each gesture corresponds to a command such as increasing or decreasing the volume, jumping forward or backward in a video, or pausing the video.

## Dataset Description

The dataset consists of videos categorized into five gesture classes, with each video divided into 30 frames. The data is organized into `train` and `val` folders, each containing subfolders for individual videos. Each subfolder contains 30 frames of a gesture. The CSV files (`train.csv` and `val.csv`) provide information on the video subfolders, gesture names, and numeric labels.

## Data Preprocessing

Preprocessing involves:
- Loading and shuffling the data.
- Cropping images to make them square.
- Resizing images to a standard dimension.
- Normalizing pixel values.
- Optionally augmenting images by applying random transformations like shifts and crops.

## DataLoader Class

- **Purpose**: Load and shuffle training and validation data from CSV files.
- **Methods**:
  - `__init__()`: Initializes the class.
  - `load_data(data_dir)`: Loads and shuffles data from the specified directory.
  - `get_data(data_dir)`: Returns the shuffled training and validation data.

## DataGenerator Class

- **Purpose**: Generate batches of preprocessed images and labels for training and validation.
- **Parameters**: Includes paths, batch size, sequence length, image dimensions, number of labels, and various flags for debugging and augmentation.
- **Methods**:
  - `__init__()`: Initializes the class with specified parameters.
  - `process_images(folder_name, img_indices)`: Processes and optionally augments images from a given folder.
  - `generator()`: Generates batches of image sequences and their corresponding labels, optionally performing ablation (stratified sampling).

## ModelManager Class

- **Purpose**: Build, compile, and train the gesture recognition model.
- **Methods**:
  - `__init__(output_path, input_shape, num_labels)`: Initializes the class with model output path, input shape, and number of labels.
  - `build(model_name, learning_rate)`: Dynamically imports the model architecture and builds the model.
  - `train(train_generator, val_generator, num_train_steps, num_val_steps, num_epochs, ...)`: Trains the model using the specified parameters and callbacks for learning rate scheduling and checkpointing.

## Trainer Class

- **Purpose**: Manage the entire training process, including data preparation, model building, and training.
- **Parameters**: Includes batch size, sequence length, image dimensions, output path, model name, learning rate, number of epochs, and various flags for debugging and augmentation.
- **Methods**:
  - `__init__(...)`: Initializes the class with specified parameters.
  - `get_docs()`: Loads and returns training and validation documents.
  - `get_num_labels(train_doc, val_doc)`: Determines the number of unique gesture labels.
  - `get_num_steps(train_doc, val_doc)`: Calculates the number of training and validation steps per epoch.
  - `build()`: Builds the model using `ModelManager`.
  - `train()`: Trains the model using `DataGenerator` for generating batches and `ModelManager` for managing the training process.
  - `plot_training_history()`: Plots the training and validation accuracy and loss over epochs.

## CustomLearningRateScheduler Class

- **Purpose**: Custom callback to adjust the learning rate during training.
- **Methods**:
  - `__init__(...)`: Initializes the class with parameters for learning rate adjustment.
  - `on_epoch_begin(epoch, logs)`: Adjusts the learning rate at the beginning of each epoch based on the specified schedule.

## Model Architecture Files (`model_*.py`)

Each `model_*.py` file defines a different model architecture using the `build_model` function. The `ModelManager` class dynamically loads these models based on the provided model name.

- **Common Components**:
  - **Imports**: Necessary TensorFlow/Keras modules for building the model.
  - **build_model(input_shape, num_labels, learning_rate)**:
    - Defines the network architecture.
    - Compiles the model with an optimizer and loss function.
    - Returns the compiled model.

- **Example: `model_15.py`**:
  - Uses MobileNet for feature extraction from each frame (2D CNN).
  - Processes the sequence of feature vectors using a GRU (Gated Recurrent Unit).
  - Includes additional layers for batch normalization, pooling, flattening, dense layers, and dropout for regularization.

## Ablation

Ablation in the code is used to reduce the dataset size for experimental purposes by performing stratified sampling. This helps to observe the effects of training on a smaller subset while maintaining the distribution of gesture labels, ensuring the smaller dataset remains representative of the original dataset.

## Experiment Summary

| Experiment Number | Model Architecture | Result | Decision & Explanation |
|-------------------|--------------------|--------|-------------------------|
| 1 | Conv3D (ablation run) | 20 OOM Error | Reduced the batch size to a smaller value (20) |
| 2 | Conv3D (ablation run) | Training accuracy: 0.4, Validation accuracy: 0.1 | Model is not learning due to small ablation size. Increased ablation size to 300 samples. |
| 3 | Conv3D (ablation run) | Training Accuracy: 1, Validation Accuracy: 0.15 | Model is overfitting, added more convolutional layers. |
| 4 | Conv3D (ablation run) | Training Accuracy: 0.69, Validation Accuracy: 0.21 | Boosting model complexity didn't solve generalization. Arranged video frames in sequential order. |
| 5 | Conv3D (ablation run) | Training Accuracy: 0.72, Validation Accuracy: 0.21 | Some improvement, but overfitting persists. Ran for 20 epochs with entire dataset. |
| 6 | Conv3D | Training Accuracy: 0.72, Validation Accuracy: 0.21 | Improvement from 15 epochs, but validation loss fluctuates. Implemented custom learning rate scheduler. |
| 7 | Conv3D | Training Accuracy: 0.87, Validation Accuracy: 0.72 | Substantial improvement after adding learning rate scheduler. Reduced overfitting. |
| 8 | Conv3D | Training Accuracy: 0.91, Validation Accuracy: 0.83 | Working well, but overfitting persists. Increased l2 norm for dense layers to 0.1. |
| 9 | Conv3D | Training Accuracy: 0.96, Validation Accuracy: 0.88 | Better accuracy but overfitting persists. Increased kernel regularizers to 0.15. |
| 10 | Conv3D | Training Accuracy: 0.94, Validation Accuracy: 0.79 | Increased kernel regularizer didn't boost performance. Set kernel initializers to he_normal. |
| 11 | Conv3D | Training Accuracy: 0.95, Validation Accuracy: 0.85 | Consistent increase in accuracy. Added more dropouts. |
| 12 | Conv3D | Training Accuracy: 0.95, Validation Accuracy: 0.55 | Dropouts not helping. Played with temporal dimension. |
| 13 | Conv3D | Training Accuracy: 0.80, Validation Accuracy: 0.86 | Reduced overfitting by changing maxpooling's temporal dimension. Ran for 50 epochs. |
| 14 | Conv3D | Training Accuracy: 0.96, Validation Accuracy: 0.91 | Achieved high accuracy by adjusting MaxPooling3D layer. Replaced flatten layer with GlobalAveragePooling3D. |
| 15 | Conv3D | Training Accuracy: 0.85, Validation Accuracy: 0.85 | Good model with reduced overfitting. Ran for 50 epochs. |
| 16 | Conv3D | Training Accuracy: 0.93, Validation Accuracy: 0.91 | Best Conv3D model so far. Moving to CNN+RNN models. |
| 17 | CNN+LSTM (ablation run) | Training Accuracy: 0.38, Validation Accuracy: 0.38 | Too much oscillation. Tried SGD. |
| 18 | CNN+LSTM (ablation run) | Training Accuracy: 0.62, Validation Accuracy: 0.52 | Reduced oscillation and improved validation accuracy with SGD and momentum. Decreased max-pooling layer units. |
| 19 | CNN+LSTM (ablation run) | Training Accuracy: 0.61, Validation Accuracy: 0.60 | Achieved good accuracy with no overfitting. Ran for 50 epochs with entire dataset. |
| 20 | CNN+LSTM | Training Accuracy: 0.99, Validation Accuracy: 0.80 | Overfitting in higher epochs. Tried GRU. |
| 21 | CNN+GRU (ablation run) | Training Accuracy: 0.72, Validation Accuracy: 0.68 | Promising model. Used entire dataset. |
| 22 | CNN+GRU | Training Accuracy: 0.93, Validation Accuracy: 0.78 | Overfitting in higher epochs. Increased dropout and added kernel regularizer. |
| 23 | CNN+GRU | Training Accuracy: 0.77, Validation Accuracy: 0.71 | Struggling with validation accuracy at higher epochs. Reduced dropouts and learning rate. |
| 24 | CNN+GRU | Training Accuracy: 0.66, Validation Accuracy: 0.68 | Slower learning but better performance. Ran for 50 epochs. |
| 25 | CNN+GRU | Training Accuracy: 0.85, Validation Accuracy: 0.77 | Promising result in lower epochs but stagnation in higher epochs. Used transfer learning with GRU. |
| 26 | CNN+GRU + Transfer Learning (SGD optimizer) | Training Accuracy: 0.99, Validation Accuracy: 0.90 | Significant improvement. Tried Adam optimizer. |
| 27 | CNN+GRU + Transfer Learning (Adam optimizer) | Training Accuracy: 0.99, Validation Accuracy: 0.98 | Best model so far with excellent performance. |

## Conclusion

The best model achieved during the experiments is a CNN+GRU architecture with transfer learning, optimized using the Adam optimizer. This model achieved a training accuracy of 0.99 and a validation accuracy of 0.98, demonstrating excellent performance and generalization capability. This model effectively recognizes the five gestures, making it suitable for deployment in smart TVs to enable gesture-based control. Future work could explore further fine-tuning and testing on a larger dataset to ensure robustness across diverse user inputs and environments.

## Environment

- **Python version**: 3.8.10
- **NumPy version**: 1.19.4
- **Skimage version**: 0.19.2
- **TensorFlow version**: 2.7.0
- **Matplotlib version**: 3.5.0
- **Scikit-learn version**: 0.24.1
