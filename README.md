# Emotion Classification with Deep Convolutional Neural Network
## Overview

This project is a deep Convolutional Neural Network (CNN) designed to classify images of human emotions into three categories: happy, sad, or angry. The CNN model is trained to recognize and distinguish these emotions from input images. This README file provides an overview of the project, its key components, and results.
## Project Components
### 1. Data Import and Preprocessing

  The project begins with importing a dataset of images containing human emotions.
  Labels corresponding to the three emotion categories (happy, sad, angry) were one-hot encoded for model training.
  Image data was preprocessed by scaling all images to a consistent format.

### 2. Data Splitting

  The dataset was split into three sets: training, validation, and test sets to train and evaluate the model effectively.

### 3. Model Architecture

  The CNN model architecture is as follows:
```python
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))
```
### 4. Model Training

  The model was trained on the training data using GPU acceleration for faster training.
  Training performance was monitored to ensure convergence.
  | ![Training Image 1](https://github.com/srdevanarayan/imageclassifier/assets/89544334/ab58fac4-4ff9-4b1d-8f76-6341b9c7615e) | ![Training Image 2](https://github.com/srdevanarayan/imageclassifier/assets/89544334/1ef15a85-4291-4cd7-93d3-b61dcf4081d0) |
|:---:|:---:|
| Loss | Accuracy |


### 5. Model Evaluation

  The model's performance was evaluated using various metrics, including Precision, Recall, and Categorical Accuracy, against the test dataset.
  The results were impressive, indicating the model's ability to accurately classify human emotions.

### 6. Prediction

  An additional image was used to demonstrate the model's ability to predict emotions from unseen data.

### 7. Model Saving

  The trained model was saved for later use, making it easier to deploy in different applications.

## Future Improvements

While the project has demonstrated success in classifying emotions, there are several ways to further improve the model's performance:
  - Collect more data to enhance the model's ability to generalize across a wider range of emotions and expressions.
  - Experiment with different CNN architectures, hyperparameters, and data augmentation techniques to optimize model performance.
  - Explore transfer learning by using pre-trained models and fine-tuning them for emotion classification.

## Conclusion

This project marks my first venture into the world of CNNs for emotion classification, and it has yielded promising results. The use of GPU acceleration has allowed for faster training, making the development process more efficient.
