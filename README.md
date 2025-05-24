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
| Model               | Accuracy | Precision | Recall | F1-Score | Remarks                                               |
|                     |          |  (Fake)   | (Fake) |  (Fake)  |                                                       |
|---------------------|----------|-----------|--------|----------|-------------------------------------------------------|
| Logistic Regression | 96.5%    | 0.98      | 0.28   | 0.44     | High precision, poor recall → misses many fake jobs   |
| Random Forest       | 97.7%    | 0.97      | 0.55   | 0.70     | Good balance                                          |
| RF + SMOTE          | 98.0%    | 0.94      | 0.63   | 0.75     | SMOTE improved recall significantly                   |
| XGBoost + SMOTE     | 97.2%    | 0.72      | 0.69   | 0.71     | Balanced but lower precision                          |
| SVM                 | 98.10%   | 0.97      | 0.62   | 0.76     | Best F1-score, top accuracy                           |
| Naive Bayes         | 96.0%    | 0.96      | 0.20   | 0.33     | Very poor recall → fails to detect most fake jobs     |

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

## 👩‍💻 Author
**Prajakta More**
EXTC Engineer | Aspiring Data Analyst & Data Scientist |
Power BI, Tableau & Python Enthusiast
Turning data into insights using ML, SQL, Excel & storytelling.



