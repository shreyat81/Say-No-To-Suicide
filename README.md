# Suicidal Ideation Detection using DistilBERT

This project implements a deep learning model to detect suicidal ideation from text data using DistilBERT, a distilled version of BERT that is faster and more efficient while maintaining good performance.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- Text classification for suicidal ideation detection
- Built with PyTorch and Hugging Face Transformers
- Includes data preprocessing and cleaning pipeline
- Model training and evaluation scripts
- Pre-trained model checkpoint available

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/suicidal-ideation-detection.git
   cd suicidal-ideation-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
```bash
python train_combined.py
```

### Making Predictions
```bash
python predict.py --text "Your text here"
```

## Dataset
The model is trained on a combination of:
- Annotated Reddit posts (`suicidal_ideation_reddit_annotated.csv`)
- Synthetic dataset (`synthetic_dataset.csv`, `synthetic_dataset_v2.csv`)

## Model Architecture
- **Base Model**: DistilBERT (uncased)
- **Classification Head**: Single linear layer
- **Sequence Length**: 256 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 10

## Training
The training script (`train_combined.py`) includes:
- Data loading and preprocessing
- Model initialization and training loop
- Learning rate scheduling
- Gradient accumulation
- Model checkpointing
- Evaluation metrics

## Inference
Use the `predict.py` script to make predictions on new text data:

```python
from predict import predict_text

result = predict_text("I'm feeling really hopeless right now...")
print(f"Suicidal probability: {result['score']:.2f}")
```

## Project Structure
```
.
├── backend/               # Backend server code
├── frontend/             # Frontend application
├── suicide-watch-model/  # Saved model checkpoints
├── generate_dataset.py   # Script for generating synthetic data
├── predict.py           # Inference script
├── requirements.txt     # Python dependencies
├── train.py             # Training script
└── train_combined.py    # Combined training script with multiple datasets
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Screenshots

![Youtube link](https://youtu.be/rVTI896q51Y)





## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Disclaimer
This project is for educational and research purposes only. It is not intended to be used as a substitute for professional medical advice, diagnosis, or treatment.
