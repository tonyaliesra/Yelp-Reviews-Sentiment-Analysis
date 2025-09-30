import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load Data
df = pd.read_csv("yelp.csv")

# 2. Select only necessary columns
df = df[['stars', 'text']]

# 3. Good (1) - Bad (0) labeling
df['label'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)

# 4. Preprocessing Function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letter characters
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

df['clean_text'] = df['text'].apply(clean_text)

# 5. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# 6. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 8. Prediction
y_pred = model.predict(X_test_vec)

# 9. Evaluation
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Predict new review
def predict_review(review_text):
    clean_review = clean_text(review_text)
    vec_review = vectorizer.transform([clean_review])
    prediction = model.predict(vec_review)[0]
    return "Good" if prediction == 1 else "Bad"

# Print confusion matrix values
print(f"True Negative (TN): {cm[0][0]}")
print(f"False Positive (FP): {cm[0][1]}")
print(f"False Negative (FN): {cm[1][0]}")
print(f"True Positive (TP): {cm[1][1]}")

# Example predictions
print(predict_review("The food was absolutely amazing and service was great!"))
print(predict_review("Terrible experience. The food was cold and tasteless."))
