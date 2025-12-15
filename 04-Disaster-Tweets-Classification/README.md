# 🌪️ NLP Disaster Tweets Classification (LSTM)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

## 📌 Project Overview
**Goal:** Classify tweets into "Real Disasters" (1) or "Metaphors/Non-Disasters" (0) using a custom LSTM model.
<br>
**Problem Type:** Natural Language Processing (Binary Classification)
<br>
**Dataset:** [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

This project explores building a text classification pipeline from scratch using **PyTorch**, involving text cleaning, custom tokenization, and solving sequence padding challenges in Recurrent Neural Networks (RNNs).

---

## ⚙️ My Approach

### 1. Data Pipeline & Preprocessing
Raw tweets are noisy and require extensive cleaning before feeding them into a neural network:
* **Cleaning:** Used `Regex` to remove URLs, HTML tags, emojis, numbers, and punctuation.
* **Normalization:** Converted all text to lowercase.
* **Tokenization:** Split sentences into individual words/tokens.
* **Vectorization:** Created a custom **Vocabulary** to map tokens to integer indices.
* **Padding:** Standardized all tweets to a fixed sequence length of **50**.

### 2. Model Architecture (LSTM)
Designed a Recurrent Neural Network to capture the sequential nature of text:
* **Embedding Layer:** Converts integer indices into dense vectors (Dim: 64).
* **LSTM Layers:** 2 stacked LSTM layers (Hidden Dim: 256) to extract temporal features.
* **Dropout (0.5):** Applied to prevent overfitting.
* **Classifier:** Fully Connected Layer + Sigmoid Activation.

### 💡 3. Key Challenge: "The Padding Trap"
**Problem:** Initially, the model was stuck at **57% accuracy** (baseline). Since many tweets are short (10-15 words) and padding was set to 50, the LSTM was processing too many zeros, causing it to "forget" the actual content by the last time step.

**Solution (Global Max Pooling):**
Instead of taking the *last hidden state*, I implemented **Global Max Pooling** across the sequence dimension:
```python
# Before (Failed): Taking the last time step
out = lstm_out[:, -1, :]

# After (Success): Taking the maximum signal across the sequence
out, _ = torch.max(lstm_out, dim=1)
```
This allowed the model to detect key "disaster words" regardless of their position or the amount of padding.

---

## 📈 Results
* **Kaggle Public Score:** ~77%
* **Training Accuracy:** ~97%
* **Validation Accuracy:** ~79%
* **Performance:** The model successfully learned to distinguish metaphors (e.g., "This song is fire") from literal meanings.

---

## 🚀 Future Improvements
To push the score above 85%, the following steps are planned:
1.  **Dynamic Padding:** To reduce noise and training time by padding per batch instead of globally.
2.  **Transformer Models:** Implementing **BERT** or **RoBERTa** (HuggingFace) to leverage pre-trained contextual embeddings.
3.  **Data Augmentation:** Using back-translation or synonym replacement to increase dataset size.

---

## 📂 Files
* `notebook.ipynb`: Complete source code (Preprocessing -> Training -> Prediction).
* `submission.csv`: Generated predictions for Kaggle.
* `best_model.pth`: Saved model weights.

**Author:** [Mohammad Ghadiri](https://github.com/Mghadiri7)
