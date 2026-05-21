"""Stylometric feature extraction and persona management."""

import numpy as np
from scipy.stats import entropy as scipy_entropy


def extract_stylometrics(text):
    """
    Extract stylometric features from text.
    
    Returns dict with keys:
    - contraction_rate
    - hedging_density
    - length_variance
    - ttr (type-token ratio)
    - avg_sent_len
    - func_ratio
    """
    words = text.lower().split()
    wc = max(len(words), 1)
    
    # Sentence segmentation
    sents = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') 
             if s.strip()]
    sent_lens = [len(s.split()) for s in sents] or [0]
    
    # Word lists
    contractions = ["n't", "'re", "'ve", "'ll", "'d", "'m", "'s"]
    hedges = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'probably',
              'seems', 'appears', 'likely', 'generally', 'often', 'sometimes']
    func_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'shall', 'can', 'and', 'but', 'or', 'in',
                  'on', 'at', 'to', 'of', 'with', 'by', 'from', 'as', 'into'}
    
    return {
        'contraction_rate': sum(text.count(c) for c in contractions) / wc,
        'hedging_density': sum(words.count(h) for h in hedges) / wc,
        'length_variance': float(np.var(sent_lens)),
        'ttr': len(set(words)) / wc,
        'avg_sent_len': float(np.mean(sent_lens)),
        'func_ratio': sum(1 for w in words if w in func_words) / wc
    }


def persona_alignment_score(text_features, persona_profile):
    """
    Compute persona alignment using KL-divergence.
    
    Returns (alignment_score, kl_divergence).
    Higher alignment = more human-like.
    """
    keys = ['contraction_rate', 'hedging_density', 'ttr', 'func_ratio']
    eps = 1e-8
    
    p = np.array([max(text_features[k], eps) for k in keys])
    q = np.array([max(persona_profile[k], eps) for k in keys])
    
    p /= p.sum()
    q /= q.sum()
    
    kl = float(scipy_entropy(p, q))
    alignment = float(np.exp(-kl))
    
    return alignment, kl


def build_persona_profile(text_list):
    """
    Build average persona profile from a list of texts.
    
    Args:
        text_list: List of text samples (typically human-written)
    
    Returns:
        dict with averaged stylometric features
    """
    feat_list = [extract_stylometrics(text) for text in text_list]
    persona = {k: float(np.mean([f[k] for f in feat_list])) 
               for k in feat_list[0]}
    return persona
