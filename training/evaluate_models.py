import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_all_data():
    """
    Load and combine all available data sources:
    1. Positive examples from extracted_open_source_records.csv and extracted_open_source_records_v2.csv
    2. Negative examples from extracted_negative_examples.csv
    """
    print("=== Loading All Data Sources ===")
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    all_data = []
    
    # 1. Load positive examples from both open source records files
    v1_file = data_dir / "extracted_open_source_records.csv"
    v2_file = data_dir / "extracted_open_source_records_v2.csv"
    pos_dfs = []
    if v1_file.exists():
        v1_df = pd.read_csv(v1_file)
        pos_dfs.append(v1_df)
        print(f"Loaded v1 positive examples: {len(v1_df)} records")
    if v2_file.exists():
        v2_df = pd.read_csv(v2_file)
        pos_dfs.append(v2_df)
        print(f"Loaded v2 positive examples: {len(v2_df)} records")
    if pos_dfs:
        pos_df = pd.concat(pos_dfs, ignore_index=True)
        pos_df['label'] = 1
        pos_df['source_dataset'] = 'positive'
        all_data.append(pos_df)
        print(f"Total positive examples: {len(pos_df)} records")
    
    # 2. Load negative examples
    negative_file = data_dir / "extracted_negative_examples.csv"
    if negative_file.exists():
        negative_df = pd.read_csv(negative_file)
        negative_df['label'] = 0
        negative_df['source_dataset'] = 'negative'
        all_data.append(negative_df)
        print(f"Loaded negative examples: {len(negative_df)} records")
    
    if not all_data:
        raise ValueError("No data sources found!")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates based on link
    combined_df = combined_df.drop_duplicates(subset=['link'])
    
    # Ensure we have required columns
    required_cols = ['title', 'link', 'published', 'summary', 'label', 'text']
    for col in required_cols:
        if col not in combined_df.columns:
            if col == 'text':
                combined_df['text'] = combined_df['title'].fillna('') + ' ' + combined_df['summary'].fillna('')
            else:
                combined_df[col] = ''
    
    print(f"\nCombined dataset: {len(combined_df)} unique records")
    print(f"Label distribution:")
    print(combined_df['label'].value_counts().sort_index())
    print(f"\nSource dataset distribution:")
    print(combined_df['source_dataset'].value_counts())
    
    return combined_df

def load_models():
    """
    Load both the old and new models.
    """
    print("\n=== Loading Models ===")
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    dag_data_dir = script_dir.parent / "dags" / "data"  # Look in DAG data directory
    
    models = {}
    
    # Load old model - try both locations
    old_model_paths = [
        data_dir / "elephant_ivory_model_old.joblib",
        dag_data_dir / "elephant_ivory_model_old.joblib"
    ]
    
    old_model_path = None
    for path in old_model_paths:
        if path.exists():
            old_model_path = path
            break
    
    if old_model_path:
        models['old_model'] = joblib.load(old_model_path)
        print(f"Loaded old model: {old_model_path}")
    else:
        print(f"Warning: Old model not found in any location")
    
    # Load new ensemble models - try both locations
    classifier_paths = [
        data_dir / "ivory_classifier.joblib",
        dag_data_dir / "ivory_classifier.joblib"
    ]
    threshold_paths = [
        data_dir / "ivory_threshold.joblib",
        dag_data_dir / "ivory_threshold.joblib"
    ]
    vectorizer_paths = [
        data_dir / "ivory_vectorizer.joblib",
        dag_data_dir / "ivory_vectorizer.joblib"
    ]
    
    # Find the first existing path for each model
    classifier_path = next((p for p in classifier_paths if p.exists()), None)
    threshold_path = next((p for p in threshold_paths if p.exists()), None)
    vectorizer_path = next((p for p in vectorizer_paths if p.exists()), None)
    
    if all(p is not None for p in [classifier_path, threshold_path, vectorizer_path]):
        models['new_classifier'] = joblib.load(classifier_path)
        models['new_threshold'] = joblib.load(threshold_path)
        models['new_vectorizer'] = joblib.load(vectorizer_path)
        print(f"Loaded new ensemble models from: {classifier_path.parent}")
    else:
        print(f"Warning: New ensemble models not found in any location")
    
    return models

def predict_with_old_model(model, texts):
    """
    Make predictions using the old model.
    """
    try:
        predictions = model.predict(texts)
        return predictions
    except Exception as e:
        print(f"Error with old model prediction: {e}")
        return None

def predict_with_new_ensemble(classifier, vectorizer, threshold, texts):
    """
    Make predictions using the new ensemble model.
    """
    try:
        # Vectorize the texts
        X = vectorizer.transform(texts)
        
        # Get classifier probabilities
        probabilities = classifier.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    except Exception as e:
        print(f"Error with new ensemble prediction: {e}")
        return None, None

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate a model and print detailed metrics.
    """
    print(f"\n=== {model_name} Evaluation ===")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                0    1")
    print(f"Actual 0    {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"       1    {cm[1,0]:4d} {cm[1,1]:4d}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def compare_models(results):
    """
    Compare the performance of both models.
    """
    print("\n=== Model Comparison ===")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    print(f"{'Metric':<12} {'Old Model':<12} {'New Model':<12} {'Difference':<12}")
    print("-" * 50)
    
    for metric in metrics:
        old_val = results.get('old_model', {}).get(metric, 0)
        new_val = results.get('new_ensemble', {}).get(metric, 0)
        diff = new_val - old_val
        
        print(f"{metric:<12} {old_val:<12.4f} {new_val:<12.4f} {diff:<+12.4f}")

def analyze_errors(y_true, y_pred_old, y_pred_new, texts, model_name):
    """
    Analyze prediction errors for a model.
    """
    print(f"\n=== {model_name} Error Analysis ===")
    
    # Find where predictions differ from true labels
    errors_old = y_true != y_pred_old
    errors_new = y_true != y_pred_new
    
    print(f"Old model errors: {errors_old.sum()}")
    print(f"New model errors: {errors_new.sum()}")
    
    # Find examples where models disagree
    disagreements = y_pred_old != y_pred_new
    print(f"Model disagreements: {disagreements.sum()}")
    
    if disagreements.sum() > 0:
        print(f"\nExamples where models disagree:")
        disagree_indices = np.where(disagreements)[0]
        for i in disagree_indices[:5]:  # Show first 5 disagreements
            print(f"\nText: {texts[i][:100]}...")
            print(f"True: {y_true[i]}, Old: {y_pred_old[i]}, New: {y_pred_new[i]}")

def save_performance_summary(results, old_report, new_report, old_cm, new_cm):
    summary_file = Path(__file__).parent / "model_performance_summary.txt"
    with open(summary_file, 'a') as f:  # Use append mode
        f.write("=== Model Performance Summary ===\n\n")
        f.write("--- Old Model ---\n")
        for k, v in results.get('old_model', {}).items():
            if k != 'confusion_matrix':
                f.write(f"{k.capitalize()}: {v}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(old_cm) + "\n")
        f.write("\nClassification Report:\n")
        f.write(old_report + "\n\n")
        f.write("--- New Ensemble Model ---\n")
        for k, v in results.get('new_ensemble', {}).items():
            if k != 'confusion_matrix':
                f.write(f"{k.capitalize()}: {v}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(new_cm) + "\n")
        f.write("\nClassification Report:\n")
        f.write(new_report + "\n")
        f.write("\n" + ("="*40) + "\n\n")  # Separator for clarity
    print(f"\nModel performance summary appended to: {summary_file}")

def main():
    """
    Main evaluation function.
    """
    print("=== Comprehensive Model Evaluation ===")
    
    # Load test set only (held-out set)
    test_set_path = Path(__file__).parent / "data" / "test_set_model.csv"
    if not test_set_path.exists():
        test_set_path = Path(__file__).parent / "data" / "test_set_classifier.csv"
    if not test_set_path.exists():
        raise FileNotFoundError("No test set found at test_set_model.csv or test_set_classifier.csv")
    data = pd.read_csv(test_set_path)
    print(f"Loaded test set: {len(data)} samples")
    X = data['text'].fillna('')
    y = data['label']
    print(f"Test set positive examples: {(y == 1).sum()}")
    print(f"Test set negative examples: {(y == 0).sum()}")

    # --- Old code: load and combine all data sources ---
    # data = load_all_data()
    # X = data['text'].fillna('')
    # y = data['label']
    # print(f"\nDataset size: {len(X)}")
    # print(f"Positive examples: {(y == 1).sum()}")
    # print(f"Negative examples: {(y == 0).sum()}")

    # Load models
    models = load_models()
    
    if not models:
        print("No models found. Exiting.")
        return
    
    # Make predictions
    results = {}
    old_report = new_report = ''
    old_cm = new_cm = None
    
    # Old model predictions
    if 'old_model' in models:
        y_pred_old = predict_with_old_model(models['old_model'], X)
        if y_pred_old is not None:
            metrics = evaluate_model(y, y_pred_old, "Old Model")
            results['old_model'] = metrics
            old_cm = metrics['confusion_matrix']
            old_report = classification_report(y, y_pred_old, target_names=['Negative', 'Positive'])
    
    # New ensemble predictions
    if all(key in models for key in ['new_classifier', 'new_vectorizer', 'new_threshold']):
        y_pred_new, probabilities = predict_with_new_ensemble(
            models['new_classifier'], 
            models['new_vectorizer'], 
            models['new_threshold'], 
            X
        )
        if y_pred_new is not None:
            metrics = evaluate_model(y, y_pred_new, "New Ensemble")
            results['new_ensemble'] = metrics
            new_cm = metrics['confusion_matrix']
            new_report = classification_report(y, y_pred_new, target_names=['Negative', 'Positive'])
    
    # Save summary only (no predictions)
    save_performance_summary(results, old_report, new_report, old_cm, new_cm)
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main() 