#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Emotion Classification Exploration
==================================

This script explores the emotion classification dataset and visualizes results.
It can be run as a script or converted to a Jupyter notebook.
"""

# %% [markdown]
# # Emotion Classification with BERT and GPT Models
# 
# This notebook explores the emotion classification dataset and results from BERT and GPT models.

# %%
# Import libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from src.utils import plot_confusion_matrix, generate_classification_report, EMOTIONS
except ImportError:
    # Define fallback EMOTIONS if the module cannot be imported
    EMOTIONS = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

# %% [markdown]
# ## 1. Load and Explore the Dataset

# %%
# Load the emotion dataset
dataset = load_dataset("emotion")
print(dataset)

# %%
# Convert to pandas DataFrames
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# %%
# Get dataset information
print(f"Train set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

# %%
# Look at the first few examples
train_df.head()

# %%
# Display emotion label mapping
for i, emotion in EMOTIONS.items():
    print(f"{i}: {emotion}")

# %% [markdown]
# ## 2. Dataset Statistics

# %%
# Combine all data for overall statistics
all_data = pd.concat([train_df, val_df, test_df])

# Count labels
label_counts = all_data["label"].value_counts().sort_index()

# Create a DataFrame with emotion names
label_df = pd.DataFrame({
    "emotion": [EMOTIONS[i] for i in label_counts.index],
    "count": label_counts.values,
    "percentage": (label_counts.values / len(all_data) * 100).round(2)
})

label_df

# %%
# Plot the distribution of emotions
plt.figure(figsize=(12, 6))
sns.barplot(x="emotion", y="count", data=label_df, palette="viridis")
plt.title("Distribution of Emotions in Dataset", fontsize=15)
plt.xlabel("Emotion", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)

# Add count labels on top of bars
for i, row in enumerate(label_df.itertuples()):
    plt.text(i, row.count + 100, f"{row.count}\n({row.percentage}%)", 
             ha="center", fontweight="bold")

plt.tight_layout()
plt.show()

# %%
# Analyze text length statistics
all_data["text_length"] = all_data["text"].apply(len)

print("Text Length Statistics:")
print(f"Mean: {all_data['text_length'].mean():.2f} characters")
print(f"Median: {all_data['text_length'].median()} characters")
print(f"Min: {all_data['text_length'].min()} characters")
print(f"Max: {all_data['text_length'].max()} characters")

# Plot text length distribution
plt.figure(figsize=(12, 6))
sns.histplot(all_data["text_length"], bins=50, kde=True)
plt.title("Distribution of Text Lengths", fontsize=15)
plt.xlabel("Text Length (characters)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.axvline(x=all_data["text_length"].mean(), color="red", linestyle="--", 
            label=f"Mean: {all_data['text_length'].mean():.1f}")
plt.axvline(x=all_data["text_length"].median(), color="green", linestyle="-.", 
            label=f"Median: {all_data['text_length'].median()}")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Text length by emotion
plt.figure(figsize=(12, 6))
sns.boxplot(x="label", y="text_length", data=all_data, palette="viridis")
plt.title("Text Length by Emotion", fontsize=15)
plt.xlabel("Emotion Label", fontsize=12)
plt.ylabel("Text Length (characters)", fontsize=12)

# Replace numeric labels with emotion names
plt.xticks(range(len(EMOTIONS)), [EMOTIONS[i] for i in range(len(EMOTIONS))], rotation=30)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Sample Texts for Each Emotion

# %%
# Print sample texts for each emotion
for label, emotion in EMOTIONS.items():
    samples = train_df[train_df["label"] == label].sample(3)
    
    print(f"\n{emotion.upper()} (Label {label}):")
    for i, row in enumerate(samples.itertuples(), 1):
        print(f"  {i}. {row.text}")

# %% [markdown]
# ## 4. Model Results Analysis
# 
# This section is for analyzing model results after training. Run this after you have trained the models.

# %%
# Define paths to results
bert_results_path = '../results/BERT_Emotion_Classifier_classification_report.csv'
gpt_results_path = '../results/GPT_Emotion_Classifier_classification_report.csv'

# Check if results files exist
bert_results_exist = os.path.exists(bert_results_path)
gpt_results_exist = os.path.exists(gpt_results_path)

if bert_results_exist or gpt_results_exist:
    print("Model results found! Proceeding with analysis.")
else:
    print("No model results found. Run the training and evaluation scripts first.")

# %%
# Load and analyze model results if they exist
if bert_results_exist and gpt_results_exist:
    # Load results
    bert_results = pd.read_csv(bert_results_path, index_col=0)
    gpt_results = pd.read_csv(gpt_results_path, index_col=0)
    
    # Extract metrics for emotions only (exclude averages)
    bert_emotion_results = bert_results.iloc[:6]
    gpt_emotion_results = gpt_results.iloc[:6]
    
    # Add emotion names as column
    bert_emotion_results['model'] = 'BERT'
    gpt_emotion_results['model'] = 'GPT'
    
    # Combine results
    combined_results = pd.concat([bert_emotion_results, gpt_emotion_results])
    
    # Map indices to emotion names
    combined_results['emotion'] = [EMOTIONS.get(i, i) for i in range(6)] * 2
    
    # Print overall metrics
    print("BERT Overall Metrics:")
    print(bert_results.loc[['macro avg', 'weighted avg']])
    
    print("\nGPT Overall Metrics:")
    print(gpt_results.loc[['macro avg', 'weighted avg']])
    
    # Visualize F1 scores by emotion and model
    plt.figure(figsize=(12, 6))
    sns.barplot(x='emotion', y='f1-score', hue='model', data=combined_results)
    plt.title('F1 Score Comparison by Emotion', fontsize=15)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()
    
    # Visualize precision by emotion and model
    plt.figure(figsize=(12, 6))
    sns.barplot(x='emotion', y='precision', hue='model', data=combined_results)
    plt.title('Precision Comparison by Emotion', fontsize=15)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()
    
    # Visualize recall by emotion and model
    plt.figure(figsize=(12, 6))
    sns.barplot(x='emotion', y='recall', hue='model', data=combined_results)
    plt.title('Recall Comparison by Emotion', fontsize=15)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Confusion Matrix Analysis

# %%
# Display confusion matrices if available
bert_cm_path = '../results/BERT_Emotion_Classifier_confusion_matrix.png'
gpt_cm_path = '../results/GPT_Emotion_Classifier_confusion_matrix.png'

if os.path.exists(bert_cm_path):
    print("BERT Confusion Matrix:")
    plt.figure(figsize=(10, 8))
    plt.imshow(plt.imread(bert_cm_path))
    plt.axis('off')
    plt.show()

if os.path.exists(gpt_cm_path):
    print("GPT Confusion Matrix:")
    plt.figure(figsize=(10, 8))
    plt.imshow(plt.imread(gpt_cm_path))
    plt.axis('off')
    plt.show()

# %% [markdown]
# ## 6. Conclusion and Insights
# 
# Based on the analysis above, we can draw the following conclusions about using BERT and GPT models for emotion classification:
# 
# 1. **Dataset Insights**:
#    - The dataset contains 6 emotion classes: sadness, joy, love, anger, fear, and surprise
#    - There is some class imbalance, with more samples for some emotions than others
#    - Text lengths vary across emotions, which might impact model performance
# 
# 2. **Model Performance**:
#    - (Add observations about model performance after training)
#    - Which emotions are easier/harder to classify?
#    - How do BERT and GPT compare overall?
#    - Which model is better for specific emotions?
# 
# 3. **Future Improvements**:
#    - Try different model architectures or pretrained models
#    - Address class imbalance
#    - Explore data augmentation techniques
#    - Fine-tune hyperparameters
#    - Consider ensemble methods

# %% [markdown]
# If you run this notebook directly, you'll need to convert it from a Python script:
# ```
# jupyter nbconvert --execute --to notebook --inplace emotion_classification_exploration.py
# ```

if __name__ == "__main__":
    print("This script can be converted to a Jupyter notebook.")
    print("Use: jupyter nbconvert --to notebook --execute emotion_classification_exploration.py") 