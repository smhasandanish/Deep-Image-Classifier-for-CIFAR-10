# Deep Image Classifier for CIFAR-10
## Overview

This repository contains a deep learning-based image classifier for the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to build a deep neural network that can classify these images into their respective categories.

## Dataset

The CIFAR-10 dataset consists of the following classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Model Architecture

We used a convolutional neural network (CNN) architecture to train the model. The architecture includes convolutional layers, max-pooling layers, fully connected layers, and dropout layers to prevent overfitting. The final layer uses softmax activation to predict the class labels.

## Usage

### 1. Training the Model

To train the model on your own, follow these steps:

1. Clone this repository: `git clone https://github.com/yourusername/Deep-Image-Classifier-CIFAR-10.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the Jupyter Notebook `train_classifier.ipynb` to train the model using the CIFAR-10 dataset.

### 2. Using the Pretrained Model

If you want to use the pretrained model for image classification, you can download it from the "models" directory in this repository. Use the following code to load the model and classify an image:

```python
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the pretrained model
model = load_model('models/cifar10_model.h5')

# Preprocess your image (replace 'image_path' with the path to your image file)
img = image.load_img('image_path', target_size=(32, 32))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Make predictions
predictions = model.predict(img)

class_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

predicted_label = class_labels[np.argmax(predictions)]
print(f"The image is classified as: {predicted_label}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CIFAR-10 dataset is available at [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
- Inspiration and guidance from deep learning tutorials and courses.

