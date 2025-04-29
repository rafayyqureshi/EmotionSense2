#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing utilities for emotion classification.
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Define emotion labels
EMOTIONS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def download_emotion_dataset():
    """
    Download the emotion dataset from Hugging Face.
    """
    print("Downloading emotion dataset...")
    dataset = load_dataset("emotion")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])
    
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Save to CSV
    train_df.to_csv("data/raw/train.csv", index=False)
    val_df.to_csv("data/raw/validation.csv", index=False)
    test_df.to_csv("data/raw/test.csv", index=False)
    
    print(f"Dataset downloaded. Train size: {len(train_df)}, "
          f"Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    return train_df, val_df, test_df

def preprocess_data(model_type="bert", max_length=128):
    """
    Preprocess the raw dataset for BERT or GPT models.
    
    Args:
        model_type: Either "bert" or "gpt"
        max_length: Maximum sequence length for tokenization
    """
    print(f"Preprocessing data for {model_type.upper()} model...")
    
    # Check if raw data exists, if not download it
    if not os.path.exists("data/raw/train.csv"):
        train_df, val_df, test_df = download_emotion_dataset()
    else:
        train_df = pd.read_csv("data/raw/train.csv")
        val_df = pd.read_csv("data/raw/validation.csv")
        test_df = pd.read_csv("data/raw/test.csv")
    
    # Select appropriate tokenizer based on model type
    if model_type.lower() == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif model_type.lower() == "gpt":
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        # GPT tokenizer doesn't have padding token by default
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Model type must be either 'bert' or 'gpt'")
    
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Process each dataset split
    for split, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        # Get texts and labels
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        # Tokenize all texts
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Save processed data
        output_file = f"data/processed/{split}_{model_type}.pt"
        
        import torch
        torch.save({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels),
            "texts": texts,  # Keep original texts for reference
        }, output_file)
        
        print(f"Saved {split} data to {output_file}")
    
    # Save the tokenizer for later use
    tokenizer_dir = f"models/{model_type}_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Saved tokenizer to {tokenizer_dir}")

def analyze_dataset():
    """
    Analyze the dataset and print statistics.
    """
    # Check if raw data exists, if not download it
    if not os.path.exists("data/raw/train.csv"):
        train_df, val_df, test_df = download_emotion_dataset()
    else:
        train_df = pd.read_csv("data/raw/train.csv")
        val_df = pd.read_csv("data/raw/validation.csv")
        test_df = pd.read_csv("data/raw/test.csv")
    
    # Combine all splits for overall statistics
    all_data = pd.concat([train_df, val_df, test_df])
    
    print("\nDataset Statistics:")
    print(f"Total samples: {len(all_data)}")
    
    # Label distribution
    label_counts = all_data["label"].value_counts().sort_index()
    print("\nLabel Distribution:")
    for label_id, count in label_counts.items():
        emotion = EMOTIONS[label_id]
        percentage = count / len(all_data) * 100
        print(f"{emotion} ({label_id}): {count} samples ({percentage:.2f}%)")
    
    # Text length statistics
    all_data["text_length"] = all_data["text"].apply(len)
    print("\nText Length Statistics:")
    print(f"Mean: {all_data['text_length'].mean():.2f} characters")
    print(f"Median: {all_data['text_length'].median()} characters")
    print(f"Min: {all_data['text_length'].min()} characters")
    print(f"Max: {all_data['text_length'].max()} characters")
    
    # Print some examples
    print("\nSample texts from each emotion:")
    for label_id, emotion in EMOTIONS.items():
        sample = all_data[all_data["label"] == label_id].sample(1).iloc[0]
        print(f"\n{emotion.upper()}: {sample['text']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing for emotion classification")
    parser.add_argument("--download", action="store_true", help="Download the emotion dataset")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset statistics")
    parser.add_argument(
        "--preprocess",
        choices=["bert", "gpt", "both"],
        help="Preprocess data for specified model(s)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization"
    )
    
    args = parser.parse_args()
    
    if args.download:
        download_emotion_dataset()
    
    if args.analyze:
        analyze_dataset()
    
    if args.preprocess:
        if args.preprocess == "both":
            preprocess_data("bert", args.max_length)
            preprocess_data("gpt", args.max_length)
        else:
            preprocess_data(args.preprocess, args.max_length)
    
    # If no arguments are provided, show help
    if not (args.download or args.analyze or args.preprocess):
        parser.print_help() 