# Script to split datasets into train, validation, and test sets [0.8, 0.1, 0.1]

# Imports
from datasets import load_dataset, DatasetDict
import os

# Define directories for and processing
processed_dir = "processed_datasets"

def load_or_process_datasets(processed_dir=processed_dir):
    # Check if processed datasets exist locally
    if os.path.exists(f"{processed_dir}/linguistic") and os.path.exists(f"{processed_dir}/meta_math"):
        print("Loading processed datasets from local disk...")
        # Load the entire DatasetDict (not just individual splits)
        linguistic_dataset = DatasetDict.load_from_disk(f"{processed_dir}/linguistic")
        meta_math_dataset = DatasetDict.load_from_disk(f"{processed_dir}/meta_math")
        print("Loaded datasets from local disk.")
    else:
        print("Loading datasets from Hugging Face and processing them...")
        # Load the raw datasets from Hugging Face, with caching handled automatically
        linguistic_dataset = load_dataset("DopeorNope/Linguistic_Calibration_Llama3.1")
        meta_math_dataset = load_dataset("meta-math/MetaMathQA")

        # Get dataset sizes and subsample to match the smallest dataset
        min_size = min(len(linguistic_dataset['train']), len(meta_math_dataset['train']))
        linguistic_dataset['train'] = linguistic_dataset['train'].shuffle(seed=42).select(range(min_size))
        meta_math_dataset['train'] = meta_math_dataset['train'].shuffle(seed=42).select(range(min_size))

        # Split the datasets into train, validation, and test (80%, 10%, 10%)
        linguistic_dataset = split_dataset(linguistic_dataset)
        meta_math_dataset = split_dataset(meta_math_dataset)

        # Save processed datasets to disk (entire DatasetDict)
        if not os.path.exists(f"{processed_dir}/linguistic"):
            linguistic_dataset.save_to_disk(f"{processed_dir}/linguistic")
        if not os.path.exists(f"{processed_dir}/meta_math"):
            meta_math_dataset.save_to_disk(f"{processed_dir}/meta_math")
        print("Datasets saved to disk.")

    return linguistic_dataset, meta_math_dataset

def split_dataset(dataset):
    # Split the dataset into 80% train, 10% validation, 10% test
    split_ratio = [0.8, 0.1, 0.1]
    train_test_split = dataset['train'].train_test_split(test_size=split_ratio[1] + split_ratio[2], seed=42)
    val_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    
    # Store the splits back into a DatasetDict
    return DatasetDict({
        'train': train_test_split['train'],
        'val': val_test_split['train'],
        'test': val_test_split['test']
    })

# Load or process the datasets
linguistic_dataset, meta_math_dataset = load_or_process_datasets(processed_dir)

