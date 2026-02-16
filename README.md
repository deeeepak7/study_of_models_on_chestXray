# 📊 Study of Models on Chest X-ray
## CNNs vs Vision Transformers for Medical Image Classification

---

## 🧩 Project Overview

This project focuses on medical image classification using deep learning–based feature extraction and classification. Both Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) are explored to analyze their effectiveness on chest X-ray images.

The experiments are conducted using **Google Colab**, with implementations in **TensorFlow** and **PyTorch**.

---

## 🎯 Objectives

- Perform medical image classification using deep learning models
- Compare CNN-based architectures with Vision Transformers (ViT)
- Analyze performance, execution time, and model complexity
- Build a reproducible and research-oriented pipeline

---

## 🧠 Models Used

### 🔹 CNN Architectures

The following pretrained CNN models (ImageNet weights) are used:

- **EfficientNet**
- **ResNet50**
- **ResNet101**
- **DenseNet121**
- **VGG19**
- **MobileNet**

These models are used for:
- Feature extraction
- Fine-tuning (where applicable)

### 🔹 Vision Transformer (ViT)

- **ViT (Vision Transformer)** implemented using PyTorch
- Captures global image dependencies, unlike CNNs which focus on local spatial features

---

## 🗂 Dataset Structure

The dataset follows a train–test split with class-wise folders:

```
dataset_root/
│
├── normal/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── test/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
│
└── tb/
    ├── train/
    │   ├── img1.jpg
    │   └── ...
    └── test/
        ├── img1.jpg
        └── ...
```

**Notes:**
- Each folder represents a class
- Compatible with:
  - TensorFlow `ImageDataGenerator`
  - PyTorch `DataLoader`
- Designed for binary medical image classification

---

## 🔄 Preprocessing Pipeline

- Image resizing according to model input requirements
- Normalization using pretrained model standards
- Conversion to tensors (for PyTorch models)
- Batch loading for efficient training
- No aggressive data augmentation to preserve medical image features

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **Classification Report**

⏱️ **Training and inference time** are also monitored.

---

## 🛠 Tech Stack & Tools

- **Python**
- **Google Colab**
- **TensorFlow / Keras**
- **PyTorch**
- **Vision Transformer (ViT)**
- **NumPy**
- **Matplotlib**
- **Scikit-learn** (metrics only)
- **GitHub**

---

## 🚀 How to Run the Project

1. Open the notebook in **Google Colab**
2. Mount Google Drive or upload the dataset
3. Ensure the dataset follows the correct folder structure
4. Select the desired model (CNN or ViT)
5. Run all cells sequentially
6. Analyze metrics and visualizations

---

## 📈 Key Insights

- CNNs perform well for local feature extraction in medical images
- Vision Transformers capture global context effectively
- Model size directly impacts training and inference time
- Proper preprocessing is critical for stable and reliable performance
- Deep models can achieve strong results even with limited medical data

---
