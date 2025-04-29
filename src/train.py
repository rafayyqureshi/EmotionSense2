#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for emotion classification models.
"""

import os
import argparse
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bert_classifier import BertForEmotionClassification
from models.gpt_classifier import GPTForEmotionClassification
from src.utils import set_seed, get_device, load_data, plot_training_curves

def create_dataloaders(model_type, batch_size=32):
    """
    Create PyTorch DataLoader objects for train and validation sets.
    
    Args:
        model_type: Type of model to use ('bert' or 'gpt')
        batch_size: Batch size for training
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    print(f"Creating dataloaders for {model_type} with batch size {batch_size}...")
    
    # Load train data
    train_data = load_data(model_type, 'train')
    val_data = load_data(model_type, 'validation')
    
    # Create TensorDatasets
    train_dataset = TensorDataset(
        train_data['input_ids'],
        train_data['attention_mask'],
        train_data['labels']
    )
    
    val_dataset = TensorDataset(
        val_data['input_ids'],
        val_data['attention_mask'],
        val_data['labels']
    )
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader

def train_model(model_type, output_dir, batch_size=32, num_epochs=5, learning_rate=5e-5, weight_decay=0.01, warmup_steps=0):
    """
    Train the emotion classification model.
    
    Args:
        model_type: Type of model to use ('bert' or 'gpt')
        output_dir: Directory to save the model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for the learning rate scheduler
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(model_type, batch_size)
    
    # Initialize model
    if model_type.lower() == 'bert':
        model = BertForEmotionClassification(num_labels=6)
    elif model_type.lower() == 'gpt':
        model = GPTForEmotionClassification(num_labels=6)
    else:
        raise ValueError("Model type must be 'bert' or 'gpt'")
    
    # Move model to device
    model.to(device)
    
    # Prepare optimizer and scheduler
    # Only optimize the classification head, keep the transformer layers frozen
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Calculate total number of training steps
    total_steps = len(train_dataloader) * num_epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Create progress bar
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Unpack batch
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            if model_type.lower() == 'bert':
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:  # GPT
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        epoch_train_loss = train_loss / len(train_dataloader)
        epoch_train_acc = train_correct / train_total
        
        # Store metrics
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc="Validation")
            
            for batch in progress_bar:
                # Unpack batch
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                
                # Forward pass
                if model_type.lower() == 'bert':
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:  # GPT
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        epoch_val_loss = val_loss / len(val_dataloader)
        epoch_val_acc = val_correct / val_total
        
        # Store metrics
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")
    
    # Calculate training time
    total_training_time = time.time() - start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    
    # Save model
    if model_type.lower() == 'bert':
        model.save(output_dir)
    else:  # GPT
        model.save(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    # Plot training curves
    plot_training_curves(
        train_losses, 
        val_losses, 
        train_accs, 
        val_accs,
        f"{model_type.upper()}_Emotion_Classifier",
        save_dir=os.path.join(output_dir, "../results")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bert", "gpt"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for the learning rate scheduler"
    )
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    ) 