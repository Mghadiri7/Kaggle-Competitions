# 🔢 Digit Recognizer - MNIST (PyTorch CNN)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

## 📌 Project Overview
**Goal:** Classify handwritten digits (0-9) using a Convolutional Neural Network (CNN).
<br>
**Problem Type:** Multi-Class Image Classification (Computer Vision)
<br>
**Dataset:** [Kaggle Digit Recognizer (MNIST)](https://www.kaggle.com/competitions/digit-recognizer)

This project demonstrates how to build a computer vision pipeline from scratch using **PyTorch**, handling CSV-based image data, creating custom datasets, and implementing a CNN architecture.

---

## ⚙️ My Approach

### 1. Data Pipeline (Custom Dataset)
Unlike standard MNIST implementations that load pre-processed data, this project handles raw **CSV data**:
* **Custom Dataset Class:** Created a `MNISTDataset` class inheriting from `torch.utils.data.Dataset`.
* **Reshaping:** Parsed 784-pixel flat vectors from CSV and reshaped them into `(1, 28, 28)` image tensors.
* **Preprocessing:** Normalized pixel values (0-255) to (0-1) using `torchvision.transforms`.
* **Splitting:** Used `torch.utils.data.random_split` to create a memory-efficient Validation set (80% Train / 20% Val).

### 2. Model Architecture (CNN)
Built a Convolutional Neural Network designed for spatial feature extraction:
* **Feature Extractor:**
    * **Conv Block 1:** Conv2d (1→32 filters) + ReLU + MaxPool2d.
    * **Conv Block 2:** Conv2d (32→64 filters) + ReLU + MaxPool2d.
* **Classifier:**
    * Flatten layer (converting 2D feature maps to 1D vector).
    * Linear Layer (Fully Connected).
    * **Dropout (0.5)** to prevent overfitting.
    * Output Layer (10 classes).

### 3. Training Configuration
* **Loss Function:** `CrossEntropyLoss` (standard for multi-class classification).
* **Optimizer:** `Adam` (Learning Rate: 0.001).
* **Training Loop:** Implemented a custom loop tracking loss and accuracy for both training and validation sets per epoch.

---

## 📈 Results
* **Kaggle Public Score:** **0.9877 (98.77%)**
* **Validation Accuracy:** ~99%
* **Performance:** The model achieved high accuracy within 10 epochs, demonstrating the effectiveness of CNNs over simple MLPs for image data.

---

## 🚀 How to Run

### ⚙️ Setup Instructions (Google Colab & Drive)
This notebook is optimized for **Google Colab** using GPU.

1.  **Download:** Download the dataset `.zip` file from the [competition link](https://www.kaggle.com/competitions/digit-recognizer/data).
2.  **Upload:** Upload `digit-recognizer.zip` to your Google Drive.
3.  **Mount Drive:** Run the notebook cells to mount Drive and extract data automatically.
4.  **Run Training:** Execute the training cells to train the CNN.
5.  **Submission:** The notebook will generate a `submission.csv` file.

**Author:** [Mohammad Ghadiri](https://github.com/Mghadiri7)
