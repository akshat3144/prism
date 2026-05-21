"""Metric functions for evaluation."""

import torch
import numpy as np
from scipy.stats import entropy as scipy_entropy
from bert_score import score as bert_score_fn
import textstat


def get_ai_score(text, tokenizer, model, device):
    """Returns P(AI-generated) in [0,1]. Higher = more AI-like."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits.float(), dim=-1)[0][1].item()


def compute_perplexity(text, tokenizer, lm_model, device):
    """Lower = more fluent. Computes GPT-2 perplexity."""
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    ids = enc.input_ids.to(device)
    with torch.no_grad():
        loss = lm_model(ids, labels=ids).loss
    return torch.exp(loss.float()).item()


def compute_bertscore(candidates, references, device='cpu'):
    """Returns mean F1 BERTScore."""
    P, R, F1 = bert_score_fn(
        candidates, references,
        lang='en', model_type='distilbert-base-uncased',
        device=device, verbose=False
    )
    return F1.mean().item()


def compute_fk_grade(text):
    """Flesch-Kincaid grade level (lower = easier read)."""
    return textstat.flesch_kincaid_grade(text)
