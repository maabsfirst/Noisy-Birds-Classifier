Noisy Birds Classification

This repository contains the solution for the Noisy Birds Classification task. The task involves classifying bird images into four categories: budgie, rubber duck, canary, and duckling. The dataset includes both labeled and unlabeled noisy images, and the goal is to achieve optimal accuracy using a model trained on noisy labeled data.
Table of Contents

    Project Overview
    Dataset
    Model Architecture
    Installation
    Usage
    Training
    Evaluation
    Results
    Submission
    Acknowledgments

Project Overview

The project applies deep learning techniques to classify noisy bird images into predefined categories using a ResNet18 architecture. The model uses augmentation techniques to handle the noise in the dataset and is trained with Adam optimizer and cross-entropy loss. The model also incorporates early stopping to avoid overfitting.

The final model is saved as model.pth and can be zipped for submission, following the competition's guidelines.
Dataset

The dataset used for this project consists of bird images divided into four classes:

    Budgie
    Rubber Duck
    Canary
    Duckling

Images are stored in directories corresponding to their class names, and the dataset is split into training (60%) and validation (40%) sets.

Data augmentations, such as random rotation, resizing, color jittering, and affine transformations, are applied to the training dataset to increase model generalization.
Model Architecture

The project uses a ResNet18 architecture with a fully connected layer adjusted to predict four classes. The model includes a dropout layer for regularization to prevent overfitting.

python

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 4)
        )

    def forward(self, x):
        return self.model(x)

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/noisy-birds-classification.git
cd noisy-birds-classification

Install the required dependencies:

bash

pip install -r requirements.txt

Alternatively, you can manually install the key dependencies:

bash

    pip install torch torchvision matplotlib pillow tqdm

    Ensure you have the dataset available in the correct format under the ./Noisy_birds/ directory.

Usage

    To train the model, simply run the Python script train.py:

    bash

    python train.py

    The model will automatically split the dataset into training and validation sets and train the model using the settings defined in the script.

Training

During training, the script will:

    Load the dataset and apply augmentations.
    Train the ResNet18 model using Adam optimizer.
    Use early stopping if validation loss stops improving.

Training will continue for 30 epochs unless early stopping is triggered.

bash

Epoch 1/30, Train Loss: 1.67, Val Loss: 1.04, Val Accuracy: 61.67%
...
Epoch 30/30, Train Loss: 0.01, Val Loss: 0.29, Val Accuracy: 91.67%

Evaluation

Once the model is trained, the final validation accuracy will be displayed. The model can also be evaluated on a test set by loading the model weights from model.pth and calculating accuracy on the test dataset.

python

# Load the trained model
model = Model().to(device)
model.load_state_dict(torch.load('model.pth'))

Results

The model achieved a final validation accuracy of 91.67% after 30 epochs of training. The training process showed steady improvement in both training and validation loss.

    Best Validation Accuracy: 91.67%
    Final Validation Loss: 0.29

Submission

To prepare the model for submission:

    The model weights are saved to model.pth after training.
    The submission zip file (submission.zip) is created, which contains the model.pth file.

To create the submission zip file:

bash

python train.py  # The zip will be created at the end of training

Once the training is complete, the model and relevant files will be packaged into submission.zip for submission.
Acknowledgments

    PyTorch: Used for model building, training, and evaluation.
    Torchvision: For using prebuilt model architectures like ResNet18.
    PIL: For image processing.
    TQDM: For progress bars during training.
