# Assignment 3.3: Transformer Fine-tuning for Sentiment Analysis

## Overview
This assignment focuses on fine-tuning pre-trained transformer models to perform sentiment classification. It covers the entire workflow from preparing data to evaluating the model, highlighting the effectiveness of transfer learning in NLP.

## Objectives
- Fine-tune a transformer model (such as BERT or RoBERTa) for sentiment classification tasks  
- Create custom PyTorch Dataset and DataLoader classes to handle text inputs  
- Utilize the Hugging Face Transformers library for model loading and tokenization  
- Implement robust training procedures including learning rate scheduling  
- Assess model performance with a variety of metrics  
- Visualize training dynamics and evaluation results

## Features
- **Pre-trained Model Usage**: Integrates BERT or RoBERTa from Hugging Face  
- **Custom Dataset Support**: Flexible dataset class tailored for text classification  
- **Enhanced Training Loop**: Supports validation, early stopping, and monitoring progress  
- **Thorough Evaluation**: Measures accuracy, precision, recall, F1-score, and plots confusion matrix  
- **Efficient Memory Handling**: Employs gradient accumulation to optimize GPU usage  
- **Visualization Tools**: Displays training loss/accuracy curves and confusion matrices  
- **Model Saving/Loading**: Enables persistence of fine-tuned models for future use

## Directory Structure
```
assignment-3.3-transformer-finetuning/
├── README.md
├── requirements.txt
├── fine-tune-transformer-for-sentimental-analysis.ipynb  # Main notebook
├── models/                           # Saved models directory
│   ├── best_model.pth
│   └── tokenizer/
├── data/                            # Dataset files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── results/                         # Training results and plots
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
└── utils/
    ├── __init__.py
    ├── data_utils.py               # Data preprocessing utilities
    ├── model_utils.py              # Model architecture utilities
    └── training_utils.py           # Training helper functions
```

## Installation
```bash
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- Datasets
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- NumPy

## Usage

### Jupyter Notebook
Open `fine-tune-transformer-for-sentimental-analysis.ipynb` to run the complete pipeline:

1. **Data Loading**: Load and explore sentiment analysis datasets
2. **Preprocessing**: Tokenization and data preparation
3. **Model Setup**: Load pre-trained transformer and add classification head
4. **Training**: Fine-tune the model with validation monitoring
5. **Evaluation**: Comprehensive performance analysis
6. **Inference**: Test the model on new examples

### Key Components

#### 1. Dataset Class
```python
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Custom dataset for sentiment analysis
```

#### 2. Model Architecture
```python
class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        # Transformer with classification head
```

#### 3. Training Loop
```python
def train_model(model, train_loader, val_loader, epochs=3):
    # Complete training pipeline with validation
```
## Model Configuration
- **Base Models**: BERT-base-uncased or RoBERTa-base  
- **Max Sequence Length**: 128 tokens (configurable)  
- **Batch Size**: 16 (modifiable depending on GPU capacity)  
- **Learning Rate**: 2e-5 with a linear warmup schedule  
- **Epochs**: 3 to 5, with early stopping support  
- **Dropout Rate**: 0.3 to reduce overfitting

## Supported Datasets
- **IMDb Movie Reviews**: Binary sentiment classification  
- **Stanford Sentiment Treebank (SST)**: Fine-grained sentiment labeling  
- **Amazon Product Reviews**: Multi-domain sentiment datasets  
- **Custom CSV Files**: User-provided text and label columns

## Evaluation Metrics
- **Accuracy**: Overall correctness of predictions  
- **Precision, Recall & F1-Score**: Calculated per class and as macro averages  
- **Confusion Matrix**: Provides detailed insight into classification errors  
- **Training Curves**: Visualize loss and accuracy trends over epochs  
- **Inference Speed**: Analysis of model prediction latency

## Training Capabilities
- **Gradient Accumulation**: Supports effectively larger batch sizes  
- **Learning Rate Scheduler**: Implements warmup followed by linear decay  
- **Early Stopping**: Monitors validation to avoid overfitting  
- **Checkpointing**: Saves the best-performing model states  
- **Memory Management**: Optimizes GPU resource utilization  
- **Mixed Precision Training**: Utilizes AMP for faster computation

## Visualization Outputs
The notebook produces a variety of helpful visualizations including:  
- Training and validation loss graphs  
- Accuracy curves across epochs  
- Confusion matrix heatmaps  
- Summaries of classification reports  
- Example predictions annotated with confidence scores

## Example Output
```
Epoch 1/3:
Train Loss: 0.542, Train Acc: 73.2%
Val Loss: 0.398, Val Acc: 82.1%

Epoch 2/3:
Train Loss: 0.312, Train Acc: 86.7%
Val Loss: 0.276, Val Acc: 88.9%

Final Test Accuracy: 89.3%
F1-Score: 0.891
```

## Model Inference
After training, the model can be used for inference:
```python
def predict_sentiment(text, model, tokenizer):
    # Returns sentiment prediction and confidence
```

## Advanced Features
- **Hyperparameter Tuning**: Perform grid search to find the best parameters  
- **Data Augmentation**: Apply text augmentation strategies to enhance training data  
- **Ensemble Learning**: Combine predictions from multiple models to boost accuracy  
- **Cross-Validation**: Use robust validation techniques for reliable performance estimates  
- **Transfer Learning**: Adapt pre-trained models to new or domain-specific datasets

## Troubleshooting Tips
- **GPU Memory Constraints**: Lower batch size or apply gradient accumulation  
- **Slow Training Speeds**: Utilize mixed precision training or switch to lighter models  
- **Suboptimal Performance**: Increase training data volume or fine-tune hyperparameters  
- **Overfitting Issues**: Raise dropout rate or decrease learning rate

## Possible Extensions
- Explore alternative transformer models like RoBERTa, DistilBERT, or ELECTRA  
- Experiment with various classification head architectures  
- Implement visualization of attention weights  
- Integrate advanced text preprocessing methods  
- Investigate domain adaptation and fine-tuning strategies

## References
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)  
- [BERT Original Paper](https://arxiv.org/abs/1810.04805)  
- [Best Practices for Fine-tuning BERT](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
