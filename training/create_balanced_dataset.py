import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def load_positive_examples():
    """
    Load positive examples from both v1 and v2 datasets.
    """
    print("=== Loading Positive Examples ===")
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    positive_examples = []
    
    # Load v1 positive examples
    v1_file = data_dir / "extracted_open_source_records.csv"
    if v1_file.exists():
        v1_df = pd.read_csv(v1_file)
        positive_examples.append(v1_df)
        print(f"Loaded v1 positive examples: {len(v1_df)} records")
    
    # Load v2 positive examples
    v2_file = data_dir / "extracted_open_source_records_v2.csv"
    if v2_file.exists():
        v2_df = pd.read_csv(v2_file)
        positive_examples.append(v2_df)
        print(f"Loaded v2 positive examples: {len(v2_df)} records")
    
    if not positive_examples:
        raise ValueError("No positive examples found!")
    
    # Combine all positive examples
    combined_positive = pd.concat(positive_examples, ignore_index=True)
    
    # Remove duplicates
    combined_positive = combined_positive.drop_duplicates(subset=['link'])
    print(f"Combined positive examples (after deduplication): {len(combined_positive)} records")
    
    return combined_positive

def load_negative_examples():
    """
    Load negative examples from the negative collection.
    """
    print("=== Loading Negative Examples ===")
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # Check multiple possible locations for negative examples
    possible_paths = [
        data_dir / "extracted_negative_examples.csv",  # Local training directory
        Path("../dags/data/extracted_negative_examples.csv"),  # DAG data directory
        Path("./dags/data/extracted_negative_examples.csv"),   # Alternative DAG path
    ]
    
    negative_file = None
    for path in possible_paths:
        if path.exists():
            negative_file = path
            break
    
    if negative_file is None:
        print("No negative examples file found. Creating empty dataset.")
        return pd.DataFrame(columns=['title', 'link', 'published', 'summary', 'label', 'text'])
    
    print(f"Found negative examples at: {negative_file}")
    negative_df = pd.read_csv(negative_file)
    print(f"Loaded negative examples: {len(negative_df)} records")
    
    # Print summary of negative examples
    if not negative_df.empty:
        print("\nNegative examples by source:")
        source_counts = negative_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        print("\nNegative examples by rejection reason:")
        reason_counts = negative_df['rejection_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
    
    return negative_df

def create_balanced_splits(positive_df, negative_df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create balanced train/validation/test splits.
    """
    print("=== Creating Balanced Splits ===")
    
    # Ensure we have the required columns
    required_cols = ['title', 'link', 'published', 'summary', 'label', 'text']
    
    # Clean and prepare positive examples
    positive_df = positive_df[required_cols].copy()
    positive_df['label'] = 1  # Ensure all positive examples are labeled as 1
    
    # Clean and prepare negative examples
    if not negative_df.empty:
        negative_df = negative_df[required_cols].copy()
        negative_df['label'] = 0  # Ensure all negative examples are labeled as 0
    else:
        print("Warning: No negative examples available. Using only positive examples.")
        negative_df = pd.DataFrame(columns=required_cols)
    
    print(f"Positive examples: {len(positive_df)}")
    print(f"Negative examples: {len(negative_df)}")
    
    # If we have both positive and negative examples, create balanced splits
    if not negative_df.empty and len(positive_df) > 0:
        # Determine the minimum number of examples per class
        min_examples = min(len(positive_df), len(negative_df))
        print(f"Balancing to {min_examples} examples per class")
        
        # Sample equal numbers from each class
        positive_sample = positive_df.sample(n=min_examples, random_state=random_state)
        negative_sample = negative_df.sample(n=min_examples, random_state=random_state)
        
        # Create splits for positive examples
        pos_train, pos_temp = train_test_split(
            positive_sample, test_size=test_size + val_size, random_state=random_state, stratify=positive_sample['label']
        )
        pos_val, pos_test = train_test_split(
            pos_temp, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=pos_temp['label']
        )
        
        # Create splits for negative examples
        neg_train, neg_temp = train_test_split(
            negative_sample, test_size=test_size + val_size, random_state=random_state, stratify=negative_sample['label']
        )
        neg_val, neg_test = train_test_split(
            neg_temp, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=neg_temp['label']
        )
        
        # Combine splits
        train_df = pd.concat([pos_train, neg_train], ignore_index=True)
        val_df = pd.concat([pos_val, neg_val], ignore_index=True)
        test_df = pd.concat([pos_test, neg_test], ignore_index=True)
        
    else:
        # If we only have positive examples, split them normally
        print("Using only positive examples for splits")
        train_df, temp_df = train_test_split(
            positive_df, test_size=test_size + val_size, random_state=random_state, stratify=positive_df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=temp_df['label']
        )
    
    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} samples ({train_df['label'].mean():.2%} positive)")
    print(f"Validation set: {len(val_df)} samples ({val_df['label'].mean():.2%} positive)")
    print(f"Test set: {len(test_df)} samples ({test_df['label'].mean():.2%} positive)")
    
    return train_df, val_df, test_df

def save_balanced_dataset(train_df, val_df, test_df, output_dir):
    """
    Save the balanced dataset splits.
    """
    print("=== Saving Balanced Dataset ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save individual splits
    train_df.to_csv(output_dir / "train_balanced.csv", index=False)
    val_df.to_csv(output_dir / "val_balanced.csv", index=False)
    test_df.to_csv(output_dir / "test_balanced.csv", index=False)
    
    # Save combined dataset
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(output_dir / "balanced_dataset.csv", index=False)
    
    # Create a training-ready version (just text and label)
    training_cols = ['text', 'label']
    train_ready = train_df[training_cols].copy()
    val_ready = val_df[training_cols].copy()
    test_ready = test_df[training_cols].copy()
    
    train_ready.to_csv(output_dir / "train_ready.csv", index=False)
    val_ready.to_csv(output_dir / "val_ready.csv", index=False)
    test_ready.to_csv(output_dir / "test_ready.csv", index=False)
    
    print(f"Saved balanced dataset to: {output_dir}")
    print(f"Files created:")
    print(f"  - train_balanced.csv ({len(train_df)} samples)")
    print(f"  - val_balanced.csv ({len(val_df)} samples)")
    print(f"  - test_balanced.csv ({len(test_df)} samples)")
    print(f"  - balanced_dataset.csv ({len(combined_df)} samples)")
    print(f"  - train_ready.csv (training format)")
    print(f"  - val_ready.csv (training format)")
    print(f"  - test_ready.csv (training format)")
    
    return output_dir

def print_dataset_summary(train_df, val_df, test_df):
    """
    Print a comprehensive summary of the dataset.
    """
    print("\n=== Dataset Summary ===")
    
    # Overall statistics
    total_samples = len(train_df) + len(val_df) + len(test_df)
    total_positive = train_df['label'].sum() + val_df['label'].sum() + test_df['label'].sum()
    total_negative = total_samples - total_positive
    
    print(f"Total samples: {total_samples}")
    print(f"Positive examples: {total_positive} ({total_positive/total_samples:.1%})")
    print(f"Negative examples: {total_negative} ({total_negative/total_samples:.1%})")
    
    # Split statistics
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"  - Positive: {train_df['label'].sum()} ({train_df['label'].mean():.1%})")
    print(f"  - Negative: {(train_df['label'] == 0).sum()} ({(train_df['label'] == 0).mean():.1%})")
    
    print(f"\nValidation set: {len(val_df)} samples")
    print(f"  - Positive: {val_df['label'].sum()} ({val_df['label'].mean():.1%})")
    print(f"  - Negative: {(val_df['label'] == 0).sum()} ({(val_df['label'] == 0).mean():.1%})")
    
    print(f"\nTest set: {len(test_df)} samples")
    print(f"  - Positive: {test_df['label'].sum()} ({test_df['label'].mean():.1%})")
    print(f"  - Negative: {(test_df['label'] == 0).sum()} ({(test_df['label'] == 0).mean():.1%})")
    
    # Text length statistics
    print(f"\nText length statistics:")
    all_texts = pd.concat([train_df['text'], val_df['text'], test_df['text']])
    print(f"  - Mean length: {all_texts.str.len().mean():.0f} characters")
    print(f"  - Median length: {all_texts.str.len().median():.0f} characters")
    print(f"  - Min length: {all_texts.str.len().min()} characters")
    print(f"  - Max length: {all_texts.str.len().max()} characters")

def main():
    """
    Main function to create a balanced dataset.
    """
    print("=== Creating Balanced Dataset ===")
    
    try:
        # Load positive and negative examples
        positive_df = load_positive_examples()
        negative_df = load_negative_examples()
        
        # Create balanced splits
        train_df, val_df, test_df = create_balanced_splits(positive_df, negative_df)
        
        # Save the balanced dataset
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "balanced_dataset"
        save_balanced_dataset(train_df, val_df, test_df, output_dir)
        
        # Print summary
        print_dataset_summary(train_df, val_df, test_df)
        
        print(f"\n=== Dataset Creation Complete ===")
        print(f"Balanced dataset saved to: {output_dir}")
        print("You can now use these files for training and evaluation!")
        
    except Exception as e:
        print(f"Error creating balanced dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 