#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT-based model for emotion classification.
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel

class GPTForEmotionClassification(nn.Module):
    """
    GPT model for emotion classification.
    """
    
    def __init__(self, num_labels=6, dropout_prob=0.1, model_name="openai-community/gpt2"):
        """
        Initialize the model.
        
        Args:
            num_labels: Number of emotion classes
            dropout_prob: Dropout probability
            model_name: Pretrained GPT model name
        """
        super(GPTForEmotionClassification, self).__init__()
        
        # Load pretrained GPT model
        self.gpt = GPT2Model.from_pretrained(model_name)
        
        # Ensure the GPT model knows about the padding token
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id
        
        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.gpt.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            labels: Ground truth labels
            
        Returns:
            dict: Model outputs including loss and logits
        """
        # Get GPT outputs
        # Since GPT doesn't have a pooler, we'll use the last hidden state
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the final hidden states
        sequence_output = outputs.last_hidden_state
        
        # Extract the last non-padded token representation for each sequence
        # This is different from BERT where we use the [CLS] token
        batch_size = sequence_output.size(0)
        
        # Either use the last token or the last non-padded token
        if attention_mask is not None:
            # Find the position of the last non-padded token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            # Gather the last non-padded token for each sequence
            last_hidden_states = torch.stack(
                [sequence_output[i, sequence_lengths[i], :] for i in range(batch_size)]
            )
        else:
            # If no attention mask, just use the last token
            last_hidden_states = sequence_output[:, -1, :]
        
        # Apply dropout and classification head
        pooled_output = self.dropout(last_hidden_states)
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
            'gpt_config': self.gpt.config.to_dict(),
            'num_labels': self.classifier.out_features,
            'model_name': self.gpt.config.name_or_path
        }
        
        # Save model state
        model_state = {
            'model_state_dict': self.state_dict(),
            'model_config': model_config
        }
        
        torch.save(model_state, f"{output_dir}/gpt_emotion_classifier.pt")
    
    @classmethod
    def load(cls, model_path, device=None):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            GPTForEmotionClassification: Loaded model
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