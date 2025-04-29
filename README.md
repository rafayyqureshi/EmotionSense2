# EmotionSense: Emotion Classification

EmotionSense is a powerful application for classifying emotions in text using advanced NLP models (BERT and GPT). The system detects six basic emotions: sadness, joy, love, anger, fear, and surprise.

## Features

- **Dual Model Architecture**: Uses both BERT and GPT models for emotion classification
- **Interactive Web Interface**: Built with Streamlit for easy interaction and visualization
- **Comparison Functionality**: Compare results between BERT and GPT models
- **Training Module**: Train models with custom parameters
- **Visualization**: View emotion probabilities through intuitive charts

## Setup Instructions

### System Requirements

- Python 3.8-3.10 (compatible with PyTorch)
- 4GB+ RAM recommended
- Internet connection for downloading pretrained models (first run only)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/rafayyqureshi/EmotionSense2.git
   cd EmotionSense2
   ```

2. Create a virtual environment:
   ```
   python3.10 -m venv py310_env
   source py310_env/bin/activate  # On Windows: py310_env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Training Models

Run the setup and training script to download data and train both models:
```
python setup_and_train.py --num_epochs 2
```

For more training options:
```
python setup_and_train.py --help
```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Enter text in the input area

3. Select which model to use (BERT, GPT, or both)

4. View the emotion classification results and probability distribution

## Project Structure

- `app.py`: Main Streamlit application
- `setup_and_train.py`: Script for setting up data and training models
- `src/`: Source code for training and evaluation
- `models/`: Model definitions and saved model weights
- `data/`: Directory for datasets

## Limitations

- Emotion classification is limited to six basic emotions
- Classification accuracy depends on the quality of the training data
- May not perform optimally on highly specialized or technical text

## Technical Details

- Built with PyTorch and Hugging Face Transformers
- BERT model: fine-tuned from bert-base-uncased
- GPT model: fine-tuned from gpt2
- Dataset: Emotion dataset with 20,000 labeled examples

## License

MIT License

## Acknowledgments

- Hugging Face for the Transformers library
- Streamlit for the web application framework
- The PyTorch team 