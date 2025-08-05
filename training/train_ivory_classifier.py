# train_ivory_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

# Load full dataset: combine positives and negatives

def load_full_data():
    data_dir = Path(__file__).parent / "data"
    # Load positives
    pos_file = data_dir / "extracted_positive_examples.csv"
    if not pos_file.exists():
        raise ValueError("No positive data found!")
    pos_df = pd.read_csv(pos_file)
    pos_df['label'] = 1
    # Load negatives
    neg_file = data_dir / "extracted_negative_examples_entire_article_labelled.csv"
    if not neg_file.exists():
        raise ValueError("No negative data found!")
    neg_df = pd.read_csv(neg_file)
    neg_df['label'] = 0
    # Combine
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    # Remove duplicates by link (in case negatives overlap)
    df = df.drop_duplicates(subset=['link'])
    print(f"After deduplication: {sum(df['label']==1)} positive, {sum(df['label']==0)} negative samples")
    # Drop rows missing required fields (only 'summary' is required)
    df = df.dropna(subset=['summary'])
    df['text'] = df['title'].fillna('').str.lower() + ' ' + df['summary'].fillna('').str.lower()
    return df['text'], df['label']

# 1) Load data
X, y = load_full_data()

# 2) 60/20/20 Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)  # 0.25 x 0.8 = 0.2

# Save test set for evaluation
pd.DataFrame({'text': X_test, 'label': y_test}).to_csv("../training/data/test_set_classifier.csv", index=False)
print("Saved test set to ../training/data/test_set_classifier.csv")

# 3) Build a scikit-learn Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# 4) Train the pipeline
pipeline.fit(X_train, y_train)

# 5) Evaluate
y_pred = pipeline.predict(X_val)
print(classification_report(y_val, y_pred))

# 6) Save the trained pipeline to a file
joblib.dump(pipeline, "elephant_ivory_model.joblib")
print("Model saved to elephant_ivory_model.joblib")