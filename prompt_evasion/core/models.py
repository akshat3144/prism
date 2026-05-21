"""Model loading utilities."""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    GPT2LMHeadModel, GPT2Tokenizer
)
from .. import config


def load_detector_model(model_name=config.DETECTOR_MODEL, device=config.DEVICE):
    """Load RoBERTa AI detector model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model


def load_gpt2_model(model_name=config.GPT2_MODEL, device=config.DEVICE):
    """Load GPT-2 model for generation and perplexity computation."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    
    return tokenizer, model


def print_device_info(device=config.DEVICE):
    """Print device and model loading information."""
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("Models ready for inference.")
