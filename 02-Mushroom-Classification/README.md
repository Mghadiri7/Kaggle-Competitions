# 🍄 Binary Prediction of Poisonous Mushrooms (Playground S4E8)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## 📌 Project Overview
**Goal:** Predict whether a mushroom is edible ('e') or poisonous ('p') based on its physical characteristics.
<br>
**Problem Type:** Binary Classification
<br>
**Dataset:** [Kaggle Playground Series - Season 4, Episode 8](https://www.kaggle.com/competitions/playground-series-s4e8)

This project utilizes a Deep Learning approach with **PyTorch** to handle a large dataset (over 2 million rows), implementing a custom training pipeline with manual preprocessing and model checkpointing.

---

## ⚙️ My Approach

### 1. Data Cleaning & Feature Selection
* **Handling High Missing Values:** Analyzed null percentages and dropped columns with >50% missing data (`stem-root`, `stem-surface`, `veil-type`, `veil-color`, `spore-print-color`) to reduce noise.
* **Imputation Strategy:**
    * **Categorical Features:** Filled missing values with the **Mode** (most frequent value).
    * **Numerical Features:** Filled missing values with the **Median**.

### 2. Data Preprocessing
* **Encoding:** Used `OrdinalEncoder` for categorical features to convert string data into integer format suitable for the neural network.
* **Target Mapping:** Converted the target class: `e` (edible) → 0, `p` (poisonous) → 1.
* **Scaling:** Applied `StandardScaler` to normalize input features, ensuring stable and faster convergence during training.
* **Split:** Used a stratified split (80% Train, 20% Validation) to maintain class balance.

### 3. Model Architecture (PyTorch)
Implemented a Neural Network using `nn.Sequential`:
* **Input Layer:** 15 Features (after cleaning).
* **Hidden Layer:** One dense layer with 60 neurons (4x input size) + `ReLU` activation.
* **Output Layer:** 1 neuron with `Sigmoid` activation (outputting probability between 0 and 1).
* **Loss Function:** `BCELoss` (Binary Cross Entropy).
* **Optimizer:** `SGD` with Momentum (0.9) and Learning Rate (0.0001).

### 4. Training Strategy
* **Batch Size:** 256
* **Model Checkpointing:** Implemented logic to save `best-model.pt` only when **Validation Loss** improves, ensuring the best performing weights are used for the final inference.

---

## 📈 Results
* **Validation Accuracy:** ~96%
* **Kaggle Public Score:** ~93%
* **Observations:** The model converged well with the SGD optimizer, achieving stable high accuracy on both training and validation sets without significant overfitting.

---

## 🚀 How to Run

### ⚙️ Setup Instructions (Google Colab & Drive)
This notebook is designed to run on **Google Colab** and load data directly from your **Google Drive**.

1.  **Download:** Download the dataset `.zip` file from the [competition link](https://www.kaggle.com/competitions/playground-series-s4e8/data).
2.  **Upload:** Upload the **zipped file** (do not unzip it) to your Google Drive (e.g., create a folder named `Datasets`).
3.  **Open Notebook:** Open the `.ipynb` file in Google Colab.
4.  **Mount Drive:** Run the first cell to mount your Google Drive.
5.  **Update Path:** Locate the `source_path` variable in the notebook and change it to match the location where you uploaded the zip file in your Drive.
    * Example: `source_path = '/content/drive/MyDrive/Datasets/playground-series-s4e8.zip'`

> **Note:** The notebook handles the copying, unzipping, and loading process automatically using `shutil` and `zipfile` for better performance.

**Author:** [Mohammad Ghadiri](https://github.com/Mghadiri7)
