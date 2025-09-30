# 📌 Yelp Reviews Sentiment Analysis

Customer reviews are one of the most valuable sources of feedback for businesses.  
In this project, I worked on **Yelp user reviews** with the goal of classifying them as either **Good (positive)** or **Bad (negative)**.  

The text data was preprocessed using **Natural Language Processing (NLP) techniques**, transformed into numerical form with **TF-IDF vectorization**, and finally classified with the **Naive Bayes (MultinomialNB)** model.  

---

## 📊 Dataset

- **Source:** [Yelp Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset)  
- **Content:** Yelp user reviews and star ratings  

**Columns used:**  
- `stars` ⭐ → User rating (1–5)  
- `text` 💬 → User review  

**Labeling logic:**  
- **Good (1):** 4 or 5 stars  
- **Bad (0):** 1, 2, or 3 stars  

---

## ⚙️ Technologies

- **Python** 
- **Pandas, NumPy** → Data processing  
- **NLTK** → Stopword removal & Lemmatization  
- **Scikit-learn** → Modeling, TF-IDF, metrics  
- **MultinomialNB** → Naive Bayes classifier  

---

## 🚀 Project Workflow

1. **Data Loading** → Import Yelp reviews  
2. **Labeling** → Assign positive/negative classes based on star ratings  
3. **Preprocessing**  
   - Remove punctuation and numbers  
   - Convert text to lowercase  
   - Remove stopwords  
   - Apply lemmatization  
4. **Vectorization** → Convert text into numerical features with TF-IDF  
5. **Model Training** → Train Naive Bayes (MultinomialNB) model  
6. **Evaluation** → Assess performance using Confusion Matrix & Classification Report  
7. **New Review Testing** → Instantly classify unseen user reviews  
