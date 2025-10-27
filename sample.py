# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import nltk
import re
import string

# Download NLTK resources (first run)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

STOPWORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer('english')

def clean_text(text):
    if pd.isna(text):
        return ""
    # lower
    text = text.lower()
    # remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # remove URLs and emails
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    # tokenize and remove stopwords and short tokens; apply stemming
    tokens = nltk.word_tokenize(text)
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def load_dataset(path='spam.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please provide spam.csv with columns 'label' and 'text'.")
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    # Try to find columns: label & text
    col_names = df.columns.str.lower()
    # heuristics:
    if 'label' in col_names and 'text' in col_names:
        df = df.rename(columns={df.columns[col_names.tolist().index('label')]: 'label',
                                df.columns[col_names.tolist().index('text')]: 'text'})
    else:
        # common variants
        possible_label_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'target' in c.lower()]
        possible_text_cols = [c for c in df.columns if 'text' in c.lower() or 'message' in c.lower() or 'body' in c.lower() or 'email' in c.lower()]
        if possible_label_cols and possible_text_cols:
            df = df.rename(columns={possible_label_cols[0]: 'label', possible_text_cols[0]: 'text'})
        else:
            # try common Kaggle format where first two columns are label and text
            df = df.iloc[:, :2]
            df.columns = ['label', 'text']
    # normalize labels to 0/1
    df['label'] = df['label'].astype(str).str.lower().map(lambda x: 1 if x in ['spam', '1', 'true', 'yes', 'y', 't'] else 0)
    # drop missing text
    df = df.dropna(subset=['text'])
    return df

def main():
    data_path = 'spam.csv'
    df = load_dataset(data_path)
    print(f"Loaded {len(df)} rows.")
    # preprocess
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    X = df['clean_text'].values
    y = df['label'].values.astype(int)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # vectorize
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # model
    dt = DecisionTreeClassifier(random_state=42, max_depth=25, min_samples_split=5)
    dt.fit(X_train_vec, y_train)

    # evaluate
    y_pred = dt.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print("Evaluation on test set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['ham','spam'], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # save model and vectorizer
    os.makedirs('saved_model', exist_ok=True)
    joblib.dump(dt, 'saved_model/dt_spam_model.joblib')
    joblib.dump(tfidf, 'saved_model/tfidf_vectorizer.joblib')
    print("Saved model and vectorizer to saved_model/")

if _name_ == '_main_':
    main()