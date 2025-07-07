# 🚨 Disaster Tweet Classifier NLP

This NLP project focuses on classifying tweets that describe **real disasters** versus those that don’t, using classic machine learning techniques. It demonstrates the practical use of **TF-IDF vectorization** and multiple models, with a focus on explainability and clean performance comparison.

---

## 🧠 Problem Statement

In a world where Twitter is often the first place people report disasters, the goal is to build a model that can distinguish between:
- `1` → **Disaster-related tweet**
- `0` → **Non-disaster tweet**

---

## 📊 Dataset Overview

- **Total samples**: ~11,370 tweets
- **Imbalanced classes**:
  - Non-disaster (`0`): 9256 tweets
  - Disaster (`1`): 2114 tweets
- Applied **downsampling** to balance the dataset for fairer training.

---

## 🧹 Preprocessing Steps

- Lowercasing
- Removing URLs, mentions, numbers, and special characters
- Expanding contractions (e.g., "can't" → "cannot")
- Stopword removal (excluding important ones like "not")
- Lemmatization

---

## 🔧 Techniques & Models Tried

| Model                      | Feature Type     | Result Summary             |
|---------------------------|------------------|----------------------------|
| ✅ Logistic Regression     | **TF-IDF**        | 🔥 Best overall performer  |
| Random Forest             | TF-IDF           | Lower recall, underperformed |
| XGBoost                   | TF-IDF           | Similar to RF, not better  |
| Logistic Regression       | FastText (custom-trained) | Poor — data too small for quality embeddings |

---

## ✅ Final Chosen Model: TF-IDF + Logistic Regression

### 📈 Performance (Balanced Dataset)
- **Accuracy:** 81%
- **Precision:** 84%
- **Recall:** 75%
- **F1-Score:** 79%

### Why it worked best:
- TF-IDF is ideal for short texts like tweets
- Logistic Regression handles sparse, high-dimensional data efficiently
- Tree-based models struggled with sparse TF-IDF data

---

## 🏗️ Project Structure

