#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit web app for emotion classification using BERT and GPT models.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time
import random  # For demo mode

# Set page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="EmotionSense: Emotion Classification",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Try to import torch, but provide fallback for Python 3.13 compatibility
TORCH_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    st.error("PyTorch is not available. Running in demo mode only.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules only if torch is available
if TORCH_AVAILABLE:
    try:
        from models.bert_classifier import BertForEmotionClassification
        from models.gpt_classifier import GPTForEmotionClassification
        from src.utils import get_device, EMOTIONS
    except ImportError:
        st.error("Failed to import model modules. Some functionality may be limited.")
else:
    # Define EMOTIONS for demo mode
    EMOTIONS = {
        0: "sadness",
        1: "joy", 
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

# Constants
MODEL_DIR = "models"
BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert_emotion_classifier.pt")
GPT_MODEL_PATH = os.path.join(MODEL_DIR, "gpt_emotion_classifier.pt")
BERT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "bert_tokenizer")
GPT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "gpt_tokenizer")
MAX_LENGTH = 128

# Custom functions
def load_models():
    """Load BERT and GPT models if available."""
    models = {}
    
    if not TORCH_AVAILABLE:
        st.sidebar.error("❌ PyTorch is not available with Python 3.13")
        st.sidebar.info("💡 Running in demo mode")
        return models
    
    device = get_device()
    
    # Check and load BERT model
    if os.path.exists(BERT_MODEL_PATH):
        try:
            models["bert"] = {
                "model": BertForEmotionClassification.load(BERT_MODEL_PATH, device),
                "tokenizer": AutoTokenizer.from_pretrained(BERT_TOKENIZER_PATH) if os.path.exists(BERT_TOKENIZER_PATH) else AutoTokenizer.from_pretrained("bert-base-uncased")
            }
            st.sidebar.success("✅ BERT model loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load BERT model: {str(e)}")
    else:
        st.sidebar.warning("⚠️ BERT model not found. Please train the model first.")
    
    # Check and load GPT model
    if os.path.exists(GPT_MODEL_PATH):
        try:
            models["gpt"] = {
                "model": GPTForEmotionClassification.load(GPT_MODEL_PATH, device),
                "tokenizer": AutoTokenizer.from_pretrained(GPT_TOKENIZER_PATH) if os.path.exists(GPT_TOKENIZER_PATH) else AutoTokenizer.from_pretrained("openai-community/gpt2")
            }
            # Ensure padding token is set for GPT tokenizer
            if models["gpt"]["tokenizer"].pad_token is None:
                models["gpt"]["tokenizer"].pad_token = models["gpt"]["tokenizer"].eos_token
            st.sidebar.success("✅ GPT model loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load GPT model: {str(e)}")
    else:
        st.sidebar.warning("⚠️ GPT model not found. Please train the model first.")
    
    return models

def run_command(command):
    """Run a shell command and capture output for display."""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Return the process for streaming
    return process

def classify_text(text, model_info):
    """Classify text using the provided model."""
    if not TORCH_AVAILABLE:
        # Generate random probabilities for demo mode
        probabilities = np.random.dirichlet(np.ones(len(EMOTIONS)), size=1)[0]
        predicted_class = np.argmax(probabilities)
        return predicted_class, probabilities
    
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Get device from the model parameters
    device = next(model.parameters()).device
    
    # Move tensors to the right device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get probabilities
    logits = outputs["logits"]
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # Get the predicted class
    predicted_class = int(torch.argmax(logits, dim=1).item())
    
    return predicted_class, probabilities

def plot_emotion_probabilities(probabilities, model_name):
    """Create a bar chart of emotion probabilities."""
    emotions = list(EMOTIONS.values())
    
    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        "Emotion": emotions,
        "Probability": probabilities
    })
    
    # Sort by probability in descending order
    df = df.sort_values("Probability", ascending=False)
    
    # Set color palette
    colors = sns.color_palette("viridis", len(emotions))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(x="Emotion", y="Probability", data=df, palette=colors, ax=ax)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.01,
            f"{df.iloc[i]['Probability']:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    
    # Customize the plot
    plt.title(f"{model_name} Emotion Probabilities", fontsize=15)
    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(0, 1.05)  # Add some space for labels
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    return fig

def show_training_options():
    """Show options for training models."""
    st.markdown("## Train Models")
    
    if not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch is not available with Python 3.13. Training is not possible.")
        st.info("To train models, you need to use a Python version 3.8-3.10 with compatible PyTorch.")
        st.code("# Install PyTorch with: \npip install torch==1.13.1")
        return
    
    st.warning("⚠️ No trained models found. You'll need to train the models before using the app.")
    
    st.markdown("### Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quick Training (Fast)")
        st.markdown("This will train both models with minimal epochs for quick testing.")
        quick_epochs = st.slider("Number of epochs (quick)", min_value=1, max_value=5, value=2)
        quick_train = st.button("Start Quick Training")
    
    with col2:
        st.markdown("#### Full Training (Better Results)")
        st.markdown("This will train both models with more epochs for better performance.")
        full_epochs = st.slider("Number of epochs (full)", min_value=3, max_value=10, value=5)
        full_train = st.button("Start Full Training")
    
    if quick_train or full_train:
        epochs = quick_epochs if quick_train else full_epochs
        
        # Create command
        command = f"python setup_and_train.py --num_epochs {epochs}"
        
        # Show command
        st.code(command)
        
        # Create a progress container
        progress_container = st.empty()
        progress_container.info("Starting training process...")
        
        # Create an output container
        output = st.empty()
        output.text("Initializing...")
        
        # Run the command
        process = run_command(command)
        
        # Track output
        output_text = ""
        
        # Show output in real-time
        for line in process.stdout:
            output_text += line
            output.text(output_text)
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            progress_container.success("✅ Training completed successfully!")
            st.button("Reload App", on_click=lambda: st.experimental_rerun())
        else:
            progress_container.error("❌ Training failed. Check the output for details.")

# Main app function
def main():
    # Header
    st.title("EmotionSense: Emotion Classification")
    st.markdown("""
    This app allows you to classify emotions in text using both BERT and GPT models.
    Enter your text below to see which emotions are detected!
    """)
    
    # Sidebar
    st.sidebar.title("EmotionSense")
    st.sidebar.markdown("### Model Information")
    
    # Load models
    models = load_models()
    
    # Check if PyTorch is available
    if not TORCH_AVAILABLE:
        st.warning("⚠️ Running in DEMO MODE - PyTorch is not available with Python 3.13")
        st.info("This is a simulation with random predictions. For real predictions, you need PyTorch compatible with your Python version.")
        
        # Text input for demo mode
        text_input = st.text_area(
            "Enter text to classify (demo mode):",
            value="I'm so happy today! Everything is going well and the weather is perfect.",
            height=150,
        )
        
        selected_model = st.radio(
            "Select model(s) to use:",
            options=["BERT (Demo)", "GPT (Demo)", "Compare Both (Demo)"],
            index=2,
        )
        
        # Process button for demo mode
        if st.button("Classify Emotion (Demo)"):
            st.markdown("---")
            st.subheader("Demo Classification Results")
            
            # Generate random results for demo
            if selected_model in ["BERT (Demo)", "Compare Both (Demo)"]:
                with st.spinner("Classifying with BERT (Demo)..."):
                    bert_class, bert_probs = classify_text(text_input, {})  # Empty model info for demo
                    bert_emotion = EMOTIONS[bert_class]
                    
                    # Display BERT results
                    bert_col1, bert_col2 = st.columns([1, 2])
                    with bert_col1:
                        st.markdown("### BERT Model (Demo)")
                        st.markdown(f"**Detected Emotion:** {bert_emotion}")
                        st.markdown(f"**Confidence:** {bert_probs[bert_class]:.2f}")
                    
                    with bert_col2:
                        bert_fig = plot_emotion_probabilities(bert_probs, "BERT (Demo)")
                        st.pyplot(bert_fig)
            
            # Generate random results for GPT demo
            if selected_model in ["GPT (Demo)", "Compare Both (Demo)"]:
                with st.spinner("Classifying with GPT (Demo)..."):
                    gpt_class, gpt_probs = classify_text(text_input, {})  # Empty model info for demo
                    gpt_emotion = EMOTIONS[gpt_class]
                    
                    # Display GPT results
                    gpt_col1, gpt_col2 = st.columns([1, 2])
                    with gpt_col1:
                        st.markdown("### GPT Model (Demo)")
                        st.markdown(f"**Detected Emotion:** {gpt_emotion}")
                        st.markdown(f"**Confidence:** {gpt_probs[gpt_class]:.2f}")
                    
                    with gpt_col2:
                        gpt_fig = plot_emotion_probabilities(gpt_probs, "GPT (Demo)")
                        st.pyplot(gpt_fig)
            
            # Compare results if both models are used
            if selected_model == "Compare Both (Demo)":
                st.markdown("---")
                st.subheader("Model Comparison (Demo)")
                
                # Create a comparison dataframe
                comparison_df = pd.DataFrame({
                    "Emotion": list(EMOTIONS.values()),
                    "BERT Probability": bert_probs,
                    "GPT Probability": gpt_probs
                })
                
                # Show match or mismatch
                if bert_class == gpt_class:
                    st.success(f"✅ Both models agree: **{EMOTIONS[bert_class]}**")
                else:
                    st.warning(f"⚠️ Models disagree: **BERT:** {EMOTIONS[bert_class]} vs **GPT:** {EMOTIONS[gpt_class]}")
                
                # Display comparison table
                st.dataframe(comparison_df.style.format({
                    "BERT Probability": "{:.4f}",
                    "GPT Probability": "{:.4f}"
                }))
                
                # Plot comparison chart
                fig, ax = plt.subplots(figsize=(12, 6))
                comparison_df.plot(
                    x="Emotion",
                    y=["BERT Probability", "GPT Probability"],
                    kind="bar",
                    ax=ax
                )
                plt.title("BERT vs GPT Probability Comparison (Demo)", fontsize=15)
                plt.xlabel("Emotion", fontsize=12)
                plt.ylabel("Probability", fontsize=12)
                plt.ylim(0, 1)
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                
        # Show information about the simulation
        st.markdown("---")
        st.markdown("### About Demo Mode")
        st.info("""
        **Demo Mode Information:**
        - This demo uses random values to simulate emotion classification
        - To run the actual models, you need:
          1. Python 3.8-3.10 (compatible with PyTorch)
          2. PyTorch installed (`pip install torch`)
          3. Trained models (run training scripts)
        """)
        
        # Show training info in sidebar
        show_training_options()
        return
    
    # Check if models are available
    if not models:
        show_training_options()
        return
    
    # Text input
    text_input = st.text_area(
        "Enter text to classify:",
        value="I'm so happy today! Everything is going well and the weather is perfect.",
        height=150,
    )
    
    # Model selection
    model_options = []
    if "bert" in models:
        model_options.append("BERT")
    if "gpt" in models:
        model_options.append("GPT")
    if "bert" in models and "gpt" in models:
        model_options.append("Compare Both")
    
    selected_model = st.radio(
        "Select model(s) to use:",
        options=model_options,
        index=len(model_options)-1 if model_options else 0,
    )
    
    # Process button
    if st.button("Classify Emotion"):
        if not model_options:
            st.error("❌ No models available. Please train at least one model first.")
            st.info("Run the training script with: `python src/train.py --model [bert|gpt]`")
            return
        
        st.markdown("---")
        st.subheader("Classification Results")
        
        # Classify with BERT
        if (selected_model == "BERT" or selected_model == "Compare Both") and "bert" in models:
            with st.spinner("Classifying with BERT..."):
                bert_class, bert_probs = classify_text(text_input, models["bert"])
                bert_emotion = EMOTIONS[bert_class]
                
                # Display BERT results
                bert_col1, bert_col2 = st.columns([1, 2])
                with bert_col1:
                    st.markdown("### BERT Model")
                    st.markdown(f"**Detected Emotion:** {bert_emotion}")
                    st.markdown(f"**Confidence:** {bert_probs[bert_class]:.2f}")
                
                with bert_col2:
                    bert_fig = plot_emotion_probabilities(bert_probs, "BERT")
                    st.pyplot(bert_fig)
        
        # Classify with GPT
        if (selected_model == "GPT" or selected_model == "Compare Both") and "gpt" in models:
            with st.spinner("Classifying with GPT..."):
                gpt_class, gpt_probs = classify_text(text_input, models["gpt"])
                gpt_emotion = EMOTIONS[gpt_class]
                
                # Display GPT results
                gpt_col1, gpt_col2 = st.columns([1, 2])
                with gpt_col1:
                    st.markdown("### GPT Model")
                    st.markdown(f"**Detected Emotion:** {gpt_emotion}")
                    st.markdown(f"**Confidence:** {gpt_probs[gpt_class]:.2f}")
                
                with gpt_col2:
                    gpt_fig = plot_emotion_probabilities(gpt_probs, "GPT")
                    st.pyplot(gpt_fig)
        
        # Compare results if both models are used
        if selected_model == "Compare Both" and "bert" in models and "gpt" in models:
            st.markdown("---")
            st.subheader("Model Comparison")
            
            # Create a comparison dataframe
            comparison_df = pd.DataFrame({
                "Emotion": list(EMOTIONS.values()),
                "BERT Probability": bert_probs,
                "GPT Probability": gpt_probs
            })
            
            # Show match or mismatch
            if bert_class == gpt_class:
                st.success(f"✅ Both models agree: **{EMOTIONS[bert_class]}**")
            else:
                st.warning(f"⚠️ Models disagree: **BERT:** {EMOTIONS[bert_class]} vs **GPT:** {EMOTIONS[gpt_class]}")
            
            # Display comparison table
            st.dataframe(comparison_df.style.format({
                "BERT Probability": "{:.4f}",
                "GPT Probability": "{:.4f}"
            }))
            
            # Plot comparison chart
            fig, ax = plt.subplots(figsize=(12, 6))
            comparison_df.plot(
                x="Emotion",
                y=["BERT Probability", "GPT Probability"],
                kind="bar",
                ax=ax
            )
            plt.title("BERT vs GPT Probability Comparison", fontsize=15)
            plt.xlabel("Emotion", fontsize=12)
            plt.ylabel("Probability", fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
    
    # App information
    with st.sidebar.expander("About EmotionSense", expanded=False):
        st.markdown("""
        **EmotionSense** is a tool for emotion classification using BERT and GPT models.
        
        The app detects 6 emotions:
        - Sadness
        - Joy
        - Love
        - Anger
        - Fear
        - Surprise
        
        This project was built using PyTorch and Hugging Face Transformers.
        """)
    
    # Usage guide
    with st.sidebar.expander("How to use", expanded=False):
        st.markdown("""
        1. Enter your text in the text area
        2. Select which model(s) to use
        3. Click "Classify Emotion"
        4. View the results and emotion probabilities
        
        If no models are available, you need to train them first:
        ```
        python src/train.py --model bert
        python src/train.py --model gpt
        ```
        """)
    
    # Training options in sidebar
    with st.sidebar.expander("Train New Models", expanded=False):
        st.markdown("### Train New Models")
        st.markdown("You can train new models to replace the existing ones.")
        
        train_epochs = st.slider("Number of epochs", min_value=1, max_value=10, value=3)
        train_model_type = st.radio("Select model to train", ["BERT", "GPT", "Both"])
        
        if st.button("Start Training"):
            if train_model_type == "Both":
                command = f"python setup_and_train.py --num_epochs {train_epochs}"
            else:
                model_arg = train_model_type.lower()
                command = f"python src/train.py --model {model_arg} --num_epochs {train_epochs}"
            
            st.code(command)
            
            with st.spinner(f"Training {train_model_type} model(s)..."):
                process = run_command(command)
                
                # Create a progress placeholder
                progress_text = st.empty()
                progress_text.info("Training in progress...")
                
                # Create an output box
                output_area = st.empty()
                
                # Collect and display output
                output_text = ""
                for line in process.stdout:
                    output_text += line
                    output_area.text_area("Training Output", output_text, height=300)
                
                # Wait for completion
                process.wait()
                
                if process.returncode == 0:
                    progress_text.success("✅ Training completed successfully!")
                    st.button("Reload App", on_click=lambda: st.experimental_rerun())
                else:
                    progress_text.error("❌ Training failed. Check the output for details.")

if __name__ == "__main__":
    main() 