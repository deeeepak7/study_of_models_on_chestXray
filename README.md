# study_of_models_on_chestXray
Project Overview

This project focuses on medical image classification using deep learning–based feature extraction and classification.
Both Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) are explored to analyze their effectiveness on medical images.

The experiments are conducted using Google Colab, with implementations in TensorFlow and PyTorch.

🎯 Objectives

Perform medical image classification using deep learning models

Compare CNN-based architectures with Vision Transformers

Analyze performance, execution time, and model complexity

Build a reproducible and research-oriented pipeline

🧠 Models Used
🔹 CNN Architectures

The following pretrained CNN models (ImageNet weights) are used:

EfficientNet

ResNet50

ResNet101

DenseNet121

VGG19

MobileNet

These models are used for:

Feature extraction

Fine-tuning (where applicable)

🔹 Vision Transformer

ViT (Vision Transformer) implemented using PyTorch

Used to capture global image dependencies unlike CNNs which focus on local patterns

🗂 Correct Dataset Structure

The dataset follows a train–test split with class-wise folders:

dataset_root/
│
├── train/
│   ├── normal/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │
│   └── tb/
│       ├── img1.jpg
│       ├── img2.jpg
│
└── test/
    ├── normal/
    │   ├── img1.jpg
    │
    └── tb/
        ├── img1.jpg


Each folder represents a class

Compatible with TensorFlow ImageDataGenerator and PyTorch DataLoader

Designed for binary medical classification

🔄 Preprocessing Pipeline

Image resizing according to model requirements

Normalization using pretrained model standards

Conversion to tensors (PyTorch)

Batch loading for efficient training

No aggressive augmentation to preserve medical features

📊 Evaluation Metrics

Model performance is evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Classification Report

Training and inference time are also monitored.

🛠 Tech Stack & Tools

Python

Google Colab

TensorFlow / Keras

PyTorch

Vision Transformer (ViT)

NumPy

Matplotlib

Scikit-learn (metrics only)

GitHub

🚀 How to Run the Project

Open the notebook in Google Colab

Mount Google Drive or upload the dataset

Ensure dataset follows the correct folder structure

Select the desired model (CNN or ViT)

Run all cells sequentially

Analyze metrics and visualizations

📈 Key Insights

CNNs perform well for local feature extraction in medical images

Vision Transformers capture global context effectively

Model size directly affects training and inference time

Proper preprocessing is critical for stable performance

Deep models can achieve strong results even with limited data
