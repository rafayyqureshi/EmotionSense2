.PHONY: install download train run-app evaluate clean all

# Set default targets
all: install download train evaluate

# Install dependencies
install:
	pip install -r requirements.txt

# Download and process data
download:
	python src/data_processing.py --download --analyze --preprocess both

# Train the models
train:
	python src/train.py --model bert --num_epochs 3
	python src/train.py --model gpt --num_epochs 3

# Evaluate both models
evaluate:
	python src/evaluate.py compare --bert_model models/bert_emotion_classifier.pt --gpt_model models/gpt_emotion_classifier.pt

# Run the web app
run-app:
	streamlit run app.py

# Quick setup - does everything in one command
quick-setup:
	python setup_and_train.py --num_epochs 3

# Clean up - remove all data and models
clean:
	rm -rf data/raw/* data/processed/* models/*.pt models/*_tokenizer

# Show help
help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make download     - Download and process dataset"
	@echo "  make train        - Train both BERT and GPT models"
	@echo "  make evaluate     - Evaluate and compare both models"
	@echo "  make run-app      - Run the Streamlit web app"
	@echo "  make quick-setup  - Run everything in sequence"
	@echo "  make clean        - Clean up data and models"
	@echo "  make all          - Run install, download, train, and evaluate" 