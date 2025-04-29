#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup and training script for emotion classification.
This script will:
1. Download and preprocess the emotion dataset
2. Train both BERT and GPT models
3. Verify the models are ready for use
"""

import os
import argparse
import subprocess
import time
import sys

def run_command(command, description):
    """Run a command and display output."""
    # Replace 'python' with 'python3' for macOS compatibility
    command = command.replace("python ", "python3 ")
    
    print(f"\n{'='*80}")
    print(f">> {description}")
    print(f">> Running: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    duration = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f">> Completed in {duration:.2f} seconds with exit code {process.returncode}")
    print(f"{'='*80}\n")
    
    if process.returncode != 0:
        print(f"Error executing: {command}")
        return False
    
    return True

def main(args):
    """Main function to setup and train models."""
    start_time = time.time()
    
    # Step 1: Download and preprocess data for both models
    print("\n🔍 Step 1: Downloading and preprocessing data...")
    data_cmd = f"python src/data_processing.py --download --analyze --preprocess both --max_length {args.max_length}"
    if not run_command(data_cmd, "Downloading and processing data"):
        return False
    
    # Step 2: Train BERT model
    print("\n🧠 Step 2: Training BERT model...")
    bert_train_cmd = (
        f"python src/train.py --model bert --output_dir {args.output_dir} "
        f"--batch_size {args.batch_size} --num_epochs {args.num_epochs} "
        f"--learning_rate {args.learning_rate}"
    )
    if not run_command(bert_train_cmd, "Training BERT model"):
        return False
    
    # Step 3: Train GPT model
    print("\n🧠 Step 3: Training GPT model...")
    gpt_train_cmd = (
        f"python src/train.py --model gpt --output_dir {args.output_dir} "
        f"--batch_size {args.batch_size} --num_epochs {args.num_epochs} "
        f"--learning_rate {args.learning_rate}"
    )
    if not run_command(gpt_train_cmd, "Training GPT model"):
        return False
    
    # Step 4: Verify models exist
    bert_model_path = os.path.join(args.output_dir, "bert_emotion_classifier.pt")
    gpt_model_path = os.path.join(args.output_dir, "gpt_emotion_classifier.pt")
    
    if not os.path.exists(bert_model_path):
        print(f"❌ Error: BERT model not found at {bert_model_path}")
        return False
    
    if not os.path.exists(gpt_model_path):
        print(f"❌ Error: GPT model not found at {gpt_model_path}")
        return False
    
    # Step 5: Compare models
    print("\n📊 Step 5: Comparing models...")
    compare_cmd = f"python src/evaluate.py compare --bert_model {bert_model_path} --gpt_model {gpt_model_path}"
    if not run_command(compare_cmd, "Comparing BERT and GPT models"):
        return False
    
    # All done!
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n✅ All done! Setup and training completed successfully.")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\nYou can now run the Streamlit app with:")
    print(f"streamlit run app.py")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and train emotion classification models")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the models"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,  # Using a smaller number for quicker training
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    success = main(args)
    sys.exit(0 if success else 1) 