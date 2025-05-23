# 🎯 Clickbait Career? Exposing Fake Jobs with AI

Fake job postings are on the rise — luring job seekers into scams and data theft. This project aims to **detect fraudulent job listings using machine learning** models, text processing, and class balancing techniques.

---

## 📌 Problem Statement

**Objective:**  
Build a machine learning classification model that predicts whether a job posting is **real** or **fake**, based on features like job description, title, location, and company profile.

---

## 📊 Dataset Overview

- **Source**: [Kaggle Fake Job Postings Dataset](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)
- **Total records**: ~18,000
- **Target Variable**: `fraudulent` (0 = Real, 1 = Fake)
- **Features**:
  - `title`, `company_profile`, `description`, `requirements`, `industry`, `location`, etc.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Removed null values and irrelevant columns
- Encoded categorical data
- Text cleaning:
  - Lowercasing
  - Stopwords removal
  - Lemmatization
- TF-IDF vectorization for textual fields

### 2. Handling Class Imbalance
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** on training data to balance classes

### 3. Model Building
Built and evaluated the following models:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Naive Bayes

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🧠 Model Comparison

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 91%      | 89%       | 93%    | 91%      |
| Random Forest       | 97%      | 95%       | 97%    | 96%      |
| XGBoost             | 96%      | 94%       | 96%    | 95%      |
| SVM (Linear)        | 90%      | 88%       | 91%    | 89%      |
| **Naive Bayes**     | 96%      | 96%       | 20%    | 33%      |

> ⚠️ Naive Bayes gave high accuracy but **failed to detect most fake jobs (low recall = 20%)**, making it unreliable in fraud detection.

---

## 🏆 Best Performing Model

**✅ Random Forest Classifier**
- Accuracy: 97%
- F1 Score: 96%
- Balanced performance on both real and fake job postings
- Works well even with limited feature engineering

---

## 📚 Future Scope

- Apply **advanced NLP models** like BERT or DistilBERT for deeper understanding of job descriptions
- Build a **Deep Learning LSTM model** for sequence-based text analysis
- Deploy as a **web app using Flask or Streamlit**
- Develop a **browser extension** to flag fake jobs in real time

---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Sklearn, Imbalanced-learn)
- NLP: NLTK, TfidfVectorizer
- Machine Learning: Logistic Regression, Random Forest, XGBoost, SVM, Naive Bayes
- Balancing: SMOTE
- Visualization: Matplotlib, Seaborn

---



