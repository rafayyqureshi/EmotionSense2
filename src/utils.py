#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for emotion classification project.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datasets import load_dataset

# Define emotion labels
EMOTIONS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get the device to use for training (GPU or CPU).
    
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon
    else:
        return torch.device("cpu")

def load_data(model_type, split):
    """
    Load preprocessed data for a specific model type and split.
    
    Args:
        model_type: Either "bert" or "gpt"
        split: Dataset split ("train", "validation", or "test")
    
    Returns:
        dict: Dictionary containing data tensors
    """
    file_path = f"data/processed/{split}_{model_type}.pt"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file {file_path} not found. "
            f"Run data_processing.py first to preprocess the data."
        )
    
    return torch.load(file_path)

def create_emotion_label_mapping():
    """
    Create mapping dictionaries between emotion IDs and names.
    
    Returns:
        tuple: (id_to_emotion, emotion_to_id) mappings
    """
    # Load the dataset to get the label mapping
    try:
        dataset_info = load_dataset("emotion")
        features = dataset_info["train"].features
        if "label" in features and hasattr(features["label"], "names"):
            id_to_emotion = {i: name for i, name in enumerate(features["label"].names)}
            emotion_to_id = {name: i for i, name in id_to_emotion.items()}
            return id_to_emotion, emotion_to_id
    except:
        pass
    
    # Fallback to predefined mapping
    id_to_emotion = EMOTIONS
    emotion_to_id = {emotion: id for id, emotion in EMOTIONS.items()}
    return id_to_emotion, emotion_to_id

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name, save_dir="results"):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_training_curves.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="results"):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get emotion mapping
    id_to_emotion, _ = create_emotion_label_mapping()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to percentage (normalize by true labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create labels for the plot
    class_names = [id_to_emotion[i] for i in sorted(id_to_emotion.keys())]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt=".1f", 
        cmap="Blues", 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_confusion_matrix.png")
    plt.close()

def generate_classification_report(y_true, y_pred, model_name, save_dir="results"):
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save reports
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get emotion mapping
    id_to_emotion, _ = create_emotion_label_mapping()
    
    # Generate classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=[id_to_emotion[i] for i in sorted(id_to_emotion.keys())],
        output_dict=True
    )
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Save to CSV
    report_df.to_csv(f"{save_dir}/{model_name}_classification_report.csv")
    
    # Also return as string for printing
    return classification_report(
        y_true, 
        y_pred, 
        target_names=[id_to_emotion[i] for i in sorted(id_to_emotion.keys())]
    )

def format_time(seconds):
    """
    Format time in seconds to hours, minutes, seconds.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}" 