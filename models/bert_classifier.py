#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERT-based model for emotion classification.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BertForEmotionClassification(nn.Module):
    """
    BERT model for emotion classification.
    """
    
    def __init__(self, num_labels=6, dropout_prob=0.1, model_name="bert-base-uncased"):
        """
        Initialize the model.
        
        Args:
            num_labels: Number of emotion classes
            dropout_prob: Dropout probability
            model_name: Pretrained BERT model name
        """
        super(BertForEmotionClassification, self).__init__()
        
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            token_type_ids: Token type IDs
            labels: Ground truth labels
            
        Returns:
            dict: Model outputs including loss and logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states
        }
    
    def save(self, output_dir):
        """
        Save the model.
        
        Args:
            output_dir: Directory to save the model
        """
        # Save model configuration
        model_config = {
            'bert_config': self.bert.config.to_dict(),
            'num_labels': self.classifier.out_features,
            'model_name': self.bert.config.name_or_path
        }
        
        # Save model state
        model_state = {
            'model_state_dict': self.state_dict(),
            'model_config': model_config
        }
        
        torch.save(model_state, f"{output_dir}/bert_emotion_classifier.pt")
    
    @classmethod
    def load(cls, model_path, device=None):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            BertForEmotionClassification: Loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Load saved state on CPU first to avoid MPS compatibility issues
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # Create model instance
        model = cls(
            num_labels=model_config['num_labels'],
            model_name=model_config['model_name']
        )
        
        # Load state dict while on CPU
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to the specified device after loading
        model = model.to(device)
        model.eval()
        
        return model 