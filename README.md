# Grok Mimic Training

This repository contains the code and setup for fine-tuning a Gemma-based model to mimic Grok's behavior using LoRA and PEFT.

## Overview

The training process involves:
- Loading a pre-trained Gemma model
- Applying LoRA fine-tuning
- Training on chat data
- Resuming from checkpoints

## Prerequisites

- Windows 10/11 with Miniconda installed at `X:\Miniconda`
- Python 3.10
- HuggingFace account with access to gated models (if using Gemma)

## Setup

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd grok-mimic-training
   ```

2. **Run environment setup**
   ```bash
   python setup_environment.py
   ```

   Or manually:
   ```bash
   conda create -n smallmodel python=3.10
   conda activate smallmodel
   pip install transformers sentencepiece datasets peft accelerate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install bitsandbytes
   ```

3. **Authenticate with HuggingFace**
   ```bash
   huggingface-cli login
   ```
   Or set environment variable:
   ```bash
   set HF_TOKEN=your_huggingface_token_here
   ```

## Training

Run the training script:
```bash
python train.py
```

The script will:
- Load the base model and tokenizer
- Apply LoRA configuration
- Load and tokenize the dataset
- Train the model with checkpoint resuming
- Save the fine-tuned model

## Configuration

Key parameters in `train.py`:
- `base_model_name`: Model to fine-tune
- `finetuned_path`: Path to existing checkpoint (optional)
- `data_path`: Path to training data (JSON format)
- Training arguments in `TrainingArguments`

## Data Format

Training data should be in JSON format with a "text" field containing the chat conversations.

## Troubleshooting

### Tokenizer Conversion Error
If you encounter:
```
ValueError: Converting from SentencePiece and Tiktoken failed
```

The code already includes fixes:
- `use_fast=False` in tokenizer loading
- Token authentication

### Missing tokenizer.model
Ensure the model repository includes the SentencePiece tokenizer files.

### CUDA Issues
The code automatically detects CUDA availability. For CPU-only training, ensure PyTorch CPU version is installed.

## Outputs

- Fine-tuned model saved to `output_dir`
- Training logs and checkpoints
- Test generation output
