import pandas as pd
import numpy as np
from pathlib import Path

def load_and_combine_datasets():
    """
    Load both extracted datasets and combine them, removing duplicates.
    """
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # Load both datasets
    print("Loading extracted datasets...")
    
    # Load v1 dataset
    v1_file = data_dir / "extracted_open_source_records.csv"
    if v1_file.exists():
        v1_df = pd.read_csv(v1_file)
        print(f"Loaded v1 dataset: {len(v1_df)} records")
    else:
        print("Warning: v1 dataset not found")
        v1_df = pd.DataFrame()
    
    # Load v2 dataset
    v2_file = data_dir / "extracted_open_source_records_v2.csv"
    if v2_file.exists():
        v2_df = pd.read_csv(v2_file)
        print(f"Loaded v2 dataset: {len(v2_df)} records")
    else:
        print("Warning: v2 dataset not found")
        v2_df = pd.DataFrame()
    
    # Combine datasets
    if not v1_df.empty and not v2_df.empty:
        combined_df = pd.concat([v1_df, v2_df], ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} records")
    elif not v1_df.empty:
        combined_df = v1_df
        print("Using only v1 dataset")
    elif not v2_df.empty:
        combined_df = v2_df
        print("Using only v2 dataset")
    else:
        raise ValueError("No datasets found!")
    
    return combined_df

def remove_duplicates(df):
    """
    Remove duplicate records based on multiple criteria.
    """
    print("Removing duplicates...")
    initial_count = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    print(f"After removing exact duplicates: {len(df)} records")
    
    # Remove duplicates based on link (URL)
    df = df.drop_duplicates(subset=['link'])
    print(f"After removing link duplicates: {len(df)} records")
    
    # Remove duplicates based on summary (keep first occurrence)
    df = df.drop_duplicates(subset=['summary'])
    print(f"After removing summary duplicates: {len(df)} records")
    
    # Remove very similar summaries (optional - can be commented out if too aggressive)
    # This removes summaries that are 95% similar
    df = remove_similar_summaries(df, similarity_threshold=0.95)
    
    final_count = len(df)
    removed_count = initial_count - final_count
    print(f"Total duplicates removed: {removed_count}")
    print(f"Final dataset size: {final_count} records")
    
    return df

def remove_similar_summaries(df, similarity_threshold=0.95):
    """
    Remove summaries that are very similar to each other.
    Uses simple character-based similarity for speed.
    """
    print(f"Removing summaries with >{similarity_threshold*100}% similarity...")
    
    # Convert summaries to lowercase for comparison
    summaries = df['summary'].str.lower().fillna('')
    
    # Simple similarity check based on character overlap
    to_remove = set()
    
    for i in range(len(summaries)):
        if i in to_remove:
            continue
            
        summary1 = summaries.iloc[i]
        if len(summary1) < 50:  # Skip very short summaries
            continue
            
        for j in range(i + 1, len(summaries)):
            if j in to_remove:
                continue
                
            summary2 = summaries.iloc[j]
            if len(summary2) < 50:  # Skip very short summaries
                continue
            
            # Calculate simple similarity
            shorter = min(len(summary1), len(summary2))
            longer = max(len(summary1), len(summary2))
            
            if shorter == 0:
                continue
                
            # Count common characters
            common_chars = sum(1 for c in summary1 if c in summary2)
            similarity = common_chars / longer
            
            if similarity > similarity_threshold:
                to_remove.add(j)
    
    if to_remove:
        df = df.drop(df.index[list(to_remove)])
        print(f"Removed {len(to_remove)} similar summaries")
    
    return df

def prepare_for_training(df):
    """
    Prepare the data for training by cleaning and formatting.
    """
    print("Preparing data for training...")
    
    # Clean the data
    df = df.copy()
    
    # Remove rows with missing critical data
    initial_count = len(df)
    df = df.dropna(subset=['summary', 'label'])
    print(f"After removing rows with missing summary/label: {len(df)} records")
    
    # Ensure label is numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    print(f"After ensuring numeric labels: {len(df)} records")
    
    # Clean text fields
    df['title'] = df['title'].fillna('').astype(str)
    df['summary'] = df['summary'].fillna('').astype(str)
    df['published'] = df['published'].fillna('').astype(str)
    
    # Remove very short summaries (likely not useful for training)
    df = df[df['summary'].str.len() > 20]
    print(f"After removing very short summaries: {len(df)} records")
    
    # Create a combined text field for training
    df['text'] = df['title'].str.lower() + ' ' + df['summary'].str.lower()
    
    # Remove rows with very short combined text
    df = df[df['text'].str.len() > 50]
    print(f"After removing very short combined text: {len(df)} records")
    
    final_count = len(df)
    removed_count = initial_count - final_count
    print(f"Total records removed during cleaning: {removed_count}")
    print(f"Final training dataset size: {final_count} records")
    
    return df

def save_prepared_data(df, output_path):
    """
    Save the prepared data for training.
    """
    print(f"Saving prepared data to {output_path}")
    
    # Save the full dataset
    df.to_csv(output_path, index=False)
    
    # Also save a version with just the columns needed for training
    training_cols = ['text', 'label', 'title', 'summary', 'link']
    training_df = df[training_cols].copy()
    training_path = output_path.parent / "training_data.csv"
    training_df.to_csv(training_path, index=False)
    
    print(f"Saved full dataset: {output_path}")
    print(f"Saved training dataset: {training_path}")
    
    return training_path

def main():
    """
    Main function to combine datasets and prepare for training.
    """
    print("=== Combining and Preparing Training Data ===")
    
    # Load and combine datasets
    combined_df = load_and_combine_datasets()
    
    # Remove duplicates
    dedup_df = remove_duplicates(combined_df)
    
    # Prepare for training
    prepared_df = prepare_for_training(dedup_df)
    
    # Save prepared data
    script_dir = Path(__file__).parent
    output_path = script_dir / "data" / "labelled.csv"
    training_path = save_prepared_data(prepared_df, output_path)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total records: {len(prepared_df)}")
    print(f"Positive examples (label=1): {prepared_df['label'].sum()}")
    print(f"Negative examples (label=0): {(prepared_df['label'] == 0).sum()}")
    print(f"Positive ratio: {prepared_df['label'].mean():.2%}")
    
    # Show some examples
    print("\n=== Sample Records ===")
    print("Sample positive examples:")
    positive_samples = prepared_df[prepared_df['label'] == 1].head(2)
    for idx, row in positive_samples.iterrows():
        print(f"Title: {row['title'][:50]}...")
        print(f"Summary: {row['summary'][:100]}...")
        print(f"Link: {row['link']}")
        print("-" * 50)
    
    print(f"\nTraining data ready! Use: {training_path}")
    print("You can now run: python train_ivory_model.py")

if __name__ == "__main__":
    main() 