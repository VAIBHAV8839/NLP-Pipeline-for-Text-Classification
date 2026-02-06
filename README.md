# NLP-Pipeline-for-Text-Classification-Using-TF-IDF-Vectorization-and-Logistic-Regression

# NLP Text Classification Pipeline (TF-IDF + Logistic Regression)

This capstone project demonstrates a complete Natural Language Processing (NLP) pipeline for basic text classification.  
The system cleans raw text, converts it into numerical vectors using **TF-IDF**, and trains a **Logistic Regression** classifier to predict categories such as sentiment (positive/negative).

---

## 📌 Project Overview

This project covers the full workflow of a traditional machine learning NLP system:

- Load labeled text data
- Clean and normalize text
- Split dataset into train/test sets
- Convert text into numerical features using TF-IDF vectorization
- Train a supervised classifier using Logistic Regression
- Evaluate performance using standard metrics
- Save and reload the trained model for future predictions

---

## 🎯 Objectives

- **Preprocessing:** Clean raw text into a standardized format  
- **Vectorization:** Convert text into numeric vectors using TF-IDF  
- **Model Training:** Train a classifier using Logistic Regression  
- **Evaluation:** Measure model performance using Accuracy, Precision, Recall, F1-score  
- **Persistence:** Save and reload the trained model using Joblib  

---

## 📊 Dataset Description

The dataset contains labeled text suitable for supervised classification.

### Columns
| Column | Description |
|--------|-------------|
| text   | Raw input sentence/review |
| label  | Target class (e.g., positive/negative) |

### Example
- `"I love this product!" → positive`
- `"Worst experience ever." → negative`

---

## 🔁 Workflow (Pipeline)

### 1) Data Loading
Data can be loaded from CSV files or manual input.

### 2) Text Cleaning
The preprocessing step includes:
- Lowercasing
- Removing URLs
- Removing punctuation
- Removing digits
- Removing extra whitespace
- Regex-based cleaning

### 3) Train/Test Split
Dataset is split into:
- **80% training**
- **20% testing**

This ensures unbiased evaluation.

### 4) Vectorization (TF-IDF)
TF-IDF converts text into a sparse numeric matrix where words are weighted by importance:
- Common words get lower weight
- Rare but meaningful words get higher weight

### 5) Model Training (Logistic Regression)
A Logistic Regression classifier is trained on the TF-IDF feature matrix.

### 6) Evaluation
The model is evaluated using:
- Accuracy score
- Confusion matrix
- Precision, Recall, and F1-score

### 7) Saving the Pipeline
The trained pipeline is saved as a `.pkl` file using Joblib, allowing the model to be reused without retraining.

---

## 🧠 Why TF-IDF + Logistic Regression?

### Why TF-IDF?
- Reduces the impact of frequent but unimportant words ("the", "a", etc.)
- Highlights rare words that carry more meaning
- Works extremely well for classical NLP problems

### Why Logistic Regression?
- Fast and efficient for training and prediction
- Performs very well with high-dimensional sparse TF-IDF features
- A strong baseline for text classification tasks

---

## 🧪 Results (Example)

Typical results (dataset dependent):
- Accuracy: ~0.80
- Precision and recall values help detect class imbalance or model bias

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- NumPy
- re (Regex)
- Scikit-learn
- Joblib

---

## 🚀 How to Run

### 1) Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

pip install -r requirements.txt
