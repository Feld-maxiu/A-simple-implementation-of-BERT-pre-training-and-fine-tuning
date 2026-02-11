# MiniBERT

A simplified implementation of BERT (Bidirectional Encoder Representations from Transformers) for pre-training and fine-tuning on downstream tasks.

## Features

- **Pre-training**: Support for MLM (Masked Language Modeling) and NSP (Next Sentence Prediction)
- **Fine-tuning**: Text classification with pre-trained BERT weights
- **Flexible Configuration**: Customizable model architecture (hidden size, layers, heads, etc.)
- **Mixed Precision Training**: FP16 support for faster training
- **Easy to Use**: Simple command-line interface

## Project Structure

```
MiniBERT/
├── bert.py                  # BERT model implementation
├── data.py                  # Data loading and preprocessing
├── train_pretrain.py        # Pre-training script
├── train_classifier.py      # Classification fine-tuning script
├── test_classifier.py       # Testing and inference script
├── data/                    # Data directory
│   ├── pretrain.txt        # Pre-training data
│   ├── train.txt           # Training data
│   └── test.txt            # Test data
└── checkpoints/             # Saved model checkpoints
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MiniBERT
```

2. Install dependencies:
```bash
pip install torch numpy
```

## Usage

### Pre-training

Train BERT from scratch on your own data:

```bash
python train_pretrain.py --batch_size 32 --epochs 10 --lr 1e-4 --hidden_size 128 --num_layers 4 --num_heads 4
```

Optional arguments:
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--seq_len`: Maximum sequence length (default: 128)
- `--hidden_size`: Hidden layer dimension (default: 128)
- `--num_layers`: Number of transformer layers (default: 4)
- `--num_heads`: Number of attention heads (default: 4)
- `--intermediate_size`: Feed-forward network intermediate size (default: 512)
- `--mlm_prob`: MLM masking probability (default: 0.15)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--fp16`: Use mixed precision training

### Fine-tuning for Classification

Fine-tune a pre-trained BERT model for text classification:

```bash
python train_classifier.py --pretrained_path checkpoints/best_model.pt --batch_size 4 --epochs 10 --lr 2e-5
```

Optional arguments:
- `--pretrained_path`: Path to pre-trained model
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 2e-5)
- `--seq_len`: Maximum sequence length (default: 64)
- `--hidden_size`: Hidden layer dimension (default: 128)
- `--num_layers`: Number of transformer layers (default: 4)
- `--num_heads`: Number of attention heads (default: 4)
- `--intermediate_size`: Feed-forward network intermediate size (default: 512)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--fp16`: Use mixed precision training

### Testing and Inference

Test a trained model on test data:

```bash
python test_classifier.py --model_path checkpoints/classifier_best.pt
```

Optional arguments:
- `--model_path`: Path to trained model (required)
- `--seq_len`: Maximum sequence length (default: 64)
- `--batch_size`: Batch size (default: 4)

## Model Architecture

MiniBERT consists of:

- **Embedding Layer**: Token, Position, and Segment embeddings
- **Transformer Encoder**: Multi-layer transformer with multi-head self-attention
- **Pre-training Heads**: MLM head and NSP head
- **Classification Head**: Linear layer for downstream classification tasks

## Data Format

### Pre-training Data
Place your pre-training text data in `data/pretrain.txt` (one sentence per line)

### Classification Data
- Training data: `data/train.txt` (format: `text\tlabel`)
- Test data: `data/test.txt` (format: `text\tlabel`)

Example:
```
This movie is great!	1
I don't like this product.	0
```

## Model Checkpoints

Pre-trained and fine-tuned models are saved in the `checkpoints/` directory:
- `best_model.pt`: Best pre-trained model
- `latest_model.pt`: Latest pre-trained model
- `classifier_best.pt`: Best fine-tuned classification model

## Performance

The model performance depends on:
- Amount of training data
- Model architecture (hidden size, number of layers)
- Training hyperparameters (learning rate, batch size, epochs)

## License

MIT License

## Acknowledgments

This implementation is inspired by the original BERT paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2019)
