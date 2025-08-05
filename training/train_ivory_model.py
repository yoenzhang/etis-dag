import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
from pathlib import Path

# Load full dataset: combine positives and negatives (copied from train_ivory_classifier.py)
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

def build_pipeline(estimator):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1,2),
            min_df=5,
            max_df=0.9,
            max_features=20_000
        )),
        ('clf', estimator)
    ])

def hyperparam_search(X, y):
    # Simplified model selection for compatibility
    models = {
        'rf': RandomForestClassifier(class_weight='balanced', random_state=42),
        'lr': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    }
    param_grid = {
        'rf': {
            'clf__n_estimators': [100],
            'clf__max_depth': [None],
            'clf__min_samples_leaf': [1]
        },
        'lr': {
            'clf__C': [1.0]
        }
    }

    best_score, best_model, best_name = 0, None, None
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced CV for speed

    for name, base in models.items():
        print(f"Testing {name} model...")
        try:
            pipe = build_pipeline(base)
            grid = GridSearchCV(
                pipe,
                param_grid[name],
                cv=cv,
                scoring='f1',
                n_jobs=1,
                verbose=1
            )
            grid.fit(X, y)
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_name = name
            print(f"{name} best score: {grid.best_score_:.3f}")
        except Exception as e:
            print(f"Error with {name} model: {e}")
            continue
    
    if best_model is None:
        raise ValueError("All models failed to train!")
    
    print(f"Best model: {best_name} (F1 ≈ {best_score:.3f})")
    return best_model

def find_optimal_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:,1]
    precision, recall, thresh = precision_recall_curve(y_val, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    return thresh[best_idx], f1_scores[best_idx]

def main():
    print("Loading data...")
    X, y = load_full_data()
    print(f"Loaded {len(X)} samples with {y.sum()} positive cases")
    
    # 60/20/20 split: train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )  # 0.25 x 0.8 = 0.2
    print(f"Train: {len(X_train)} samples, Validation: {len(X_val)} samples, Test: {len(X_test)} samples")

    # 1) Hyperparameter search
    print("Starting hyperparameter search...")
    best_pipeline = hyperparam_search(X_train, y_train)
    # Print which model was selected
    best_model_name = type(best_pipeline.named_steps['clf']).__name__
    print(f"Selected model: {best_model_name}")

    # 2) Evaluate on validation set
    print("Evaluating on validation set...")
    y_pred = best_pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))

    # 3) Find best threshold
    print("Finding optimal threshold...")
    thresh, f1 = find_optimal_threshold(best_pipeline, X_val, y_val)
    print(f"Optimal threshold = {thresh:.3f} (val F1 = {f1:.3f})")
    
    # Save test set for evaluation
    test_set = pd.DataFrame({'text': X_test, 'label': y_test})
    test_set.to_csv("../training/data/test_set_model.csv", index=False)
    print("Saved test set to ../training/data/test_set_model.csv")

    joblib.dump(
        best_pipeline.named_steps['tfidf'], 
        "../dags/data/ivory_vectorizer.joblib",
        compress=3,
        protocol=4
    )
    joblib.dump(
        best_pipeline.named_steps['clf'], 
        "../dags/data/ivory_classifier.joblib",
        compress=3,
        protocol=4
    )
    joblib.dump(
        thresh, 
        "../dags/data/ivory_threshold.joblib",
        compress=3,
        protocol=4
    )

    print("Saved vectorizer, classifier, and threshold with compatibility settings.")
    
if __name__ == "__main__":
    main() 