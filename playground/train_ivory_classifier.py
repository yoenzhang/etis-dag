# train_ivory_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1) Load data
df = pd.read_csv("./data/labelled.csv")  # CSV with columns: title, link, published, summary, label

print(df.columns)

# 2) Combine title + summary as main text
df['text'] = df['title'].fillna('') + " " + df['summary'].fillna('')
X = df['text']            # Features
y = df['label']           # 1 = relevant, 0 = not relevant

# 3) Train/Test split
# stratify=y helps maintain the same proportion of 0/1 in train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Build a scikit-learn Pipeline
#    TfidfVectorizer transforms text into numerical vectors
#    LogisticRegression is our classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))  # max_iter=1000 for better convergence
])

# 5) Train the pipeline
pipeline.fit(X_train, y_train)

# 6) Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 7) Save the trained pipeline to a file
joblib.dump(pipeline, "elephant_ivory_model.joblib")
print("Model saved to elephant_ivory_model.joblib")
