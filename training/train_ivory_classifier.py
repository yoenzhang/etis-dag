import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

def load_data(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=['title','summary','label'])
    df['text'] = df['title'].str.lower() + ' ' + df['summary'].str.lower()
    return df['text'], df['label']

def build_pipeline(estimator, use_dense=False):
    if use_dense:
        # For models that require dense matrices
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                ngram_range=(1,2),
                min_df=5,
                max_df=0.9,
                max_features=20_000
            )),
            ('to_dense', lambda X: X.toarray()),
            ('clf', estimator)
        ])
    else:
        # For models that work with sparse matrices
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
    # try different model families (removed HistGradientBoosting due to sparse matrix issues)
    models = {
        'rf': (RandomForestClassifier(class_weight='balanced', random_state=42), False),
        'lr': (LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42), False)
    }
    param_grid = {
        'rf': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_leaf': [1,2]
        },
        'lr': {
            'clf__C': [0.1, 1.0, 10.0]
        }
    }

    best_score, best_model, best_name = 0, None, None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (base, use_dense) in models.items():
        print(f"Testing {name} model...")
        try:
            pipe = build_pipeline(base, use_dense)
            grid = GridSearchCV(
                pipe,
                param_grid[name],
                cv=cv,
                scoring='f1',
                n_jobs=-1,
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
    X, y = load_data("./data/labelled.csv")
    print(f"Loaded {len(X)} samples with {y.sum()} positive cases")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} samples, Validation: {len(X_val)} samples")

    # 1) Hyperparameter search
    print("Starting hyperparameter search...")
    best_pipeline = hyperparam_search(X_train, y_train)

    # 2) Evaluate on validation set
    print("Evaluating on validation set...")
    y_pred = best_pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))

    # 3) Find best threshold
    print("Finding optimal threshold...")
    thresh, f1 = find_optimal_threshold(best_pipeline, X_val, y_val)
    print(f"Optimal threshold = {thresh:.3f} (val F1 = {f1:.3f})")

    # 4) Persist artifacts separately
    print("Saving artifacts...")
    joblib.dump(best_pipeline.named_steps['tfidf'], "../dags/data/ivory_vectorizer.joblib")
    joblib.dump(best_pipeline.named_steps['clf'], "../dags/data/ivory_classifier.joblib")
    joblib.dump(thresh, "../dags/data/ivory_threshold.joblib")

    print("Saved vectorizer, classifier, and threshold.")
    
if __name__ == "__main__":
    main()
