#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for emotion classification models.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bert_classifier import BertForEmotionClassification
from models.gpt_classifier import GPTForEmotionClassification
from src.utils import (
    get_device, 
    load_data, 
    plot_confusion_matrix, 
    generate_classification_report,
    EMOTIONS
)

def evaluate_model(model_path, test_data_path=None, batch_size=32):
    """
    Evaluate a trained emotion classification model.
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test data (if None, use the default test data)
        batch_size: Batch size for evaluation
    """
    # Determine model type from path
    if "bert" in model_path.lower():
        model_type = "bert"
    elif "gpt" in model_path.lower():
        model_type = "gpt"
    else:
        raise ValueError("Cannot determine model type from path. Path should contain 'bert' or 'gpt'.")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {model_type.upper()} model from {model_path}...")
    if model_type == "bert":
        model = BertForEmotionClassification.load(model_path, device)
    else:  # GPT
        model = GPTForEmotionClassification.load(model_path, device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Load test data
    if test_data_path is None:
        print(f"Loading default test data for {model_type}...")
        test_data = load_data(model_type, 'test')
    else:
        # TODO: Handle custom test data
        raise NotImplementedError("Custom test data loading not implemented yet.")
    
    # Create DataLoader
    test_dataset = TensorDataset(
        test_data['input_ids'],
        test_data['attention_mask'],
        test_data['labels']
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize variables for evaluation
    all_preds = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    
    # Evaluate model
    print("Evaluating model...")
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Evaluation")
        
        for batch in progress_bar:
            # Unpack batch
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Forward pass
            if model_type == "bert":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:  # GPT
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Update metrics
            test_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Calculate accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions and labels for later analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate final metrics
    test_loss = test_loss / len(test_dataloader)
    test_accuracy = correct / total
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        model_name=f"{model_type.upper()}_Emotion_Classifier"
    )
    
    # Generate classification report
    print("\nGenerating classification report...")
    report = generate_classification_report(
        y_true=all_labels,
        y_pred=all_preds,
        model_name=f"{model_type.upper()}_Emotion_Classifier"
    )
    
    print("\nClassification Report:")
    print(report)
    
    # Print some example predictions
    print("\nSample Predictions:")
    sample_texts = test_data['texts']
    
    # Get a random sample of test indices
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(sample_texts), min(10, len(sample_texts)), replace=False)
    
    for idx in sample_indices:
        text = sample_texts[idx]
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        
        print(f"\nText: {text}")
        print(f"True Emotion: {EMOTIONS[true_label]}")
        print(f"Predicted Emotion: {EMOTIONS[pred_label]}")
        print(f"Correct: {'✓' if true_label == pred_label else '✗'}")

def compare_models(bert_model_path, gpt_model_path, batch_size=32):
    """
    Compare BERT and GPT models side by side.
    
    Args:
        bert_model_path: Path to the trained BERT model
        gpt_model_path: Path to the trained GPT model
        batch_size: Batch size for evaluation
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading BERT model from {bert_model_path}...")
    bert_model = BertForEmotionClassification.load(bert_model_path, device)
    bert_model.eval()
    
    print(f"Loading GPT model from {gpt_model_path}...")
    gpt_model = GPTForEmotionClassification.load(gpt_model_path, device)
    gpt_model.eval()
    
    # Load test data for both models
    bert_test_data = load_data("bert", 'test')
    gpt_test_data = load_data("gpt", 'test')
    
    # Create DataLoaders
    bert_test_dataset = TensorDataset(
        bert_test_data['input_ids'],
        bert_test_data['attention_mask'],
        bert_test_data['labels']
    )
    
    gpt_test_dataset = TensorDataset(
        gpt_test_data['input_ids'],
        gpt_test_data['attention_mask'],
        gpt_test_data['labels']
    )
    
    bert_test_dataloader = DataLoader(
        bert_test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    gpt_test_dataloader = DataLoader(
        gpt_test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Evaluate BERT model
    print("\nEvaluating BERT model...")
    bert_all_preds = []
    bert_all_labels = []
    bert_test_loss = 0
    bert_correct = 0
    bert_total = 0
    
    with torch.no_grad():
        for batch in tqdm(bert_test_dataloader, desc="BERT Evaluation"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            bert_test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            bert_correct += (preds == labels).sum().item()
            bert_total += labels.size(0)
            
            bert_all_preds.extend(preds.cpu().numpy())
            bert_all_labels.extend(labels.cpu().numpy())
    
    bert_test_loss = bert_test_loss / len(bert_test_dataloader)
    bert_test_accuracy = bert_correct / bert_total
    
    # Evaluate GPT model
    print("\nEvaluating GPT model...")
    gpt_all_preds = []
    gpt_all_labels = []
    gpt_test_loss = 0
    gpt_correct = 0
    gpt_total = 0
    
    with torch.no_grad():
        for batch in tqdm(gpt_test_dataloader, desc="GPT Evaluation"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = gpt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            gpt_test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            gpt_correct += (preds == labels).sum().item()
            gpt_total += labels.size(0)
            
            gpt_all_preds.extend(preds.cpu().numpy())
            gpt_all_labels.extend(labels.cpu().numpy())
    
    gpt_test_loss = gpt_test_loss / len(gpt_test_dataloader)
    gpt_test_accuracy = gpt_correct / gpt_total
    
    # Print comparison results
    print("\n----- Model Comparison -----")
    print(f"BERT - Loss: {bert_test_loss:.4f}, Accuracy: {bert_test_accuracy:.4f}")
    print(f"GPT  - Loss: {gpt_test_loss:.4f}, Accuracy: {gpt_test_accuracy:.4f}")
    
    # Generate classification reports
    print("\nBERT Classification Report:")
    bert_report = generate_classification_report(
        bert_all_labels,
        bert_all_preds,
        "BERT_Emotion_Classifier"
    )
    print(bert_report)
    
    print("\nGPT Classification Report:")
    gpt_report = generate_classification_report(
        gpt_all_labels,
        gpt_all_preds,
        "GPT_Emotion_Classifier"
    )
    print(gpt_report)
    
    # Generate confusion matrices
    plot_confusion_matrix(
        bert_all_labels,
        bert_all_preds,
        "BERT_Emotion_Classifier"
    )
    
    plot_confusion_matrix(
        gpt_all_labels,
        gpt_all_preds,
        "GPT_Emotion_Classifier"
    )
    
    # Create model comparison table for each emotion
    print("\nPer-Emotion Comparison:")
    print("| Emotion   | BERT Precision | BERT Recall | BERT F1 | GPT Precision | GPT Recall | GPT F1 |")
    print("|-----------|----------------|-------------|---------|---------------|------------|--------|")
    
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calculate metrics for each model
    bert_precision, bert_recall, bert_f1, _ = precision_recall_fscore_support(
        bert_all_labels, bert_all_preds, average=None
    )
    
    gpt_precision, gpt_recall, gpt_f1, _ = precision_recall_fscore_support(
        gpt_all_labels, gpt_all_preds, average=None
    )
    
    # Print metrics for each emotion
    for i, emotion in EMOTIONS.items():
        print(f"| {emotion:<9} | {bert_precision[i]:.4f}        | {bert_recall[i]:.4f}     | {bert_f1[i]:.4f} | {gpt_precision[i]:.4f}       | {gpt_recall[i]:.4f}    | {gpt_f1[i]:.4f} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate emotion classification models")
    
    # Create subparsers for different evaluation modes
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")
    
    # Single model evaluation
    single_parser = subparsers.add_parser("single", help="Evaluate a single model")
    single_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    single_parser.add_argument(
        "--test_data",
        type=str,
        help="Path to the test data (if None, use default test data)"
    )
    single_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare BERT and GPT models")
    compare_parser.add_argument(
        "--bert_model",
        type=str,
        required=True,
        help="Path to the trained BERT model"
    )
    compare_parser.add_argument(
        "--gpt_model",
        type=str,
        required=True,
        help="Path to the trained GPT model"
    )
    compare_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        evaluate_model(
            model_path=args.model_path,
            test_data_path=args.test_data,
            batch_size=args.batch_size
        )
    elif args.mode == "compare":
        compare_models(
            bert_model_path=args.bert_model,
            gpt_model_path=args.gpt_model,
            batch_size=args.batch_size
        )
    else:
        parser.print_help() 