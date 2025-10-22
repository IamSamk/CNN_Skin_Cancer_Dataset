# Skin Cancer Classification with Deep Learning

A PyTorch-based project for binary classification of skin lesions (benign vs malignant) using dermoscopic images.
This repository contains the full training pipeline, evaluation code, and a pretrained model for direct inference.

***

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup \& Requirements](#setup--requirements)
- [Training \& Evaluation](#training--evaluation)
- [Model Inference](#model-inference)
- [Results](#results)
- [References \& Links](#references--links)

***

## Project Overview

This project implements a deep learning pipeline for classifying skin lesions as benign or malignant using dermoscopic images.
The model is based on a ResNet34 architecture, trained and evaluated on a large, curated dataset.
All code is provided in a single Jupyter notebook for reproducibility and ease of experimentation.

***

## Dataset

- **Source:** [Skin Cancer MNIST: HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Structure:** Images are organized into `train/benign`, `train/malignant`, `test/benign`, and `test/malignant` folders.
- **Preprocessing:** Images are resized, normalized, and augmented during training.

***

## Repository Structure

```
.
├── train/
│   ├── benign/
│   └── malignant/
├── test/
│   ├── benign/
│   └── malignant/
├── AIMLC3_1.ipynb         # Main Jupyter notebook (full pipeline)
├── Skin_cancer_model.pth  # Trained PyTorch model weights
└── README.md              # This file
```


***

## Setup \& Requirements

**Python version:** 3.8+
**Recommended environment:** Conda or venv

**Install dependencies:**

```bash
pip install torch torchvision matplotlib scikit-learn pillow tqdm
```


***

## Training \& Evaluation

- All code for data loading, augmentation, model training, validation, and evaluation is in `AIMLC3_1.ipynb`.
- The notebook includes:
    - Data preprocessing and augmentation
    - Model definition (ResNet34, binary output)
    - Training loop with progress bars and metrics
    - Evaluation on the test set (accuracy, precision, recall, F1, confusion matrix)
    - Batch and single-image inference utilities

***

## Model Inference

**To use the pretrained model:**

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(r"C:\Users\Samarth Kadam\Documents\Skin_cancer_model.pth", map_location='cpu'))
model.eval()

# Define transform (must match training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Predict on a single image
img = Image.open("path/to/image.jpg").convert("RGB")
img_t = test_transform(img).unsqueeze(0)
with torch.no_grad():
    output = model(img_t)
    prob = torch.sigmoid(output).item()
    pred = "malignant" if prob > 0.5 else "benign"
print(f"Prediction: {pred} (prob={prob:.3f})")
```


***

## Results

- **Best model:** ResNet34, trained for 15 epochs
- **Test accuracy:** ~94% (see confusion matrix in notebook)
- **Generalization:** Model performance may drop on images from different sources due to domain shift.

***

## References \& Links

- [Skin Cancer MNIST: HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision.models](https://pytorch.org/vision/stable/models.html)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)
- [tqdm](https://tqdm.github.io/)

***

**For questions or suggestions, open an issue or contact the maintainer.**

***

