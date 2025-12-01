# 🚢 Titanic - Machine Learning from Disaster (PyTorch Implementation)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## 📌 Project Overview
**Goal:** Predict survival on the Titanic using a Deep Learning approach with PyTorch.
<br>
**Problem Type:** Binary Classification
<br>
**Dataset:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

This project focuses on building a neural network from scratch using **PyTorch**, handling data preprocessing manually, and implementing a custom training loop with model checkpointing.

---

## ⚙️ My Approach

### 1. Exploratory Data Analysis (EDA) & Cleaning
* Analyzed the correlation between features like `Sex`, `Pclass`, and `Embarked` with survival rates.
* **Feature Selection:** Dropped high-cardinality or irrelevant columns: `Name`, `Ticket`, `Cabin`, and `PassengerId`.

### 2. Data Preprocessing
* **Missing Values Imputation:**
    * `Age`: Filled with the **mean** of the training set.
    * `Embarked`: Filled with the **mode** (most frequent value).
    * *Note:* Strictly calculated statistics on the **Train set only** to prevent data leakage.
* **Encoding:**
    * `Sex`: Mapped to binary (0 for Male, 1 for Female).
    * `Embarked`: Used **One-Hot Encoding** (creating dummy variables).
* **Scaling:** Applied `StandardScaler` to normalize all input features for better neural network performance.

### 3. Model Architecture (PyTorch)
Implemented a **Multi-Layer Perceptron (MLP)** using `nn.Sequential`:
* **Input Layer:** Matching the number of features.
* **Hidden Layers:** Two dense layers (64 and 32 neurons) with **ReLU** activation.
* **Regularization:** Applied **Dropout (0.3)** after each hidden layer to prevent overfitting.
* **Output Layer:** 1 neuron (Logits).
* **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy).
* **Optimizer:** `Adam` (Learning Rate: 0.001).

### 4. Training Strategy
* **Split:** 80% Training, 20% Validation.
* **Model Checkpointing:** Implemented a logic to save the `best_model.pt` only when **Validation Loss** decreased, preventing overfitting in later epochs.
* **Batching:** Used `DataLoader` and `TensorDataset` for efficient mini-batch training.

---

## 📈 Results
* **Kaggle Public Score:** ~0.78
* **Observations:** The model achieved ~82% accuracy on the validation set during training before overfitting started (which was handled by loading the best saved weights).

---

## 🚀 How to Run

### ⚙️ Setup Instructions (Google Colab & Drive)
This notebook is designed to run on **Google Colab** and load data directly from your **Google Drive**.

1.  **Download:** Download the dataset `.zip` file from the [competition link](https://www.kaggle.com/c/titanic).
2.  **Upload:** Upload the **zipped file** (do not unzip it) to your Google Drive (e.g., create a folder named `Datasets`).
3.  **Open Notebook:** Open the `.ipynb` file in Google Colab.
4.  **Mount Drive:** Run the first cell to mount your Google Drive.
5.  **Update Path:** Locate the `source_path` variable in the notebook and change it to match the location where you uploaded the zip file in your Drive.
    * Example: `source_path = '/content/drive/MyDrive/Datasets/titanic.zip'`

> **Note:** The notebook handles the copying, unzipping, and loading process automatically using `shutil` and `zipfile` for better performance.

**Author:** [Mohammad Ghadiri](https://github.com/Mghadiri7)
