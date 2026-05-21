"""Main evaluation pipeline."""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from .. import config
from ..utils.metrics import get_ai_score, compute_perplexity, compute_bertscore, compute_fk_grade
from ..utils.stylometrics import extract_stylometrics, persona_alignment_score
from . import methods


def run_evaluation(ai_df, pool_df, persona_profile, det_tok, det_model,
                   gpt2_tok, gpt2_model, device=config.DEVICE, n_eval=config.N_EVAL):
    """
    Run full evaluation on 4 methods: original, ours, SICO, SDA.
    
    Args:
        ai_df: DataFrame of AI-generated texts for evaluation
        pool_df: DataFrame with persona-aligned examples
        persona_profile: Target persona profile
        det_tok: Detector tokenizer
        det_model: Detector model
        gpt2_tok: GPT-2 tokenizer
        gpt2_model: GPT-2 model
        device: Device to run on
        n_eval: Number of samples to evaluate
    
    Returns:
        results dict: {method: [list of result dicts]}
    """
    eval_set = ai_df.head(n_eval).reset_index(drop=True)
    results = {m: [] for m in ['original', 'ours', 'sico', 'sda']}
    
    print(f"Evaluating {n_eval} samples across 4 methods...")
    
    for _, row in tqdm(eval_set.iterrows(), total=n_eval):
        ai_text = row['answer']
        question = row['question']
        
        def record(text, original=None):
            """Create a result record for a text."""
            ai_s = get_ai_score(text, det_tok, det_model, device)
            feats = extract_stylometrics(text)
            align, kl = persona_alignment_score(feats, persona_profile)
            return {
                'text': text,
                'original': original or text,
                'ai_score': ai_s,
                'evaded': ai_s < config.THRESHOLD,
                'ppl': compute_perplexity(text, gpt2_tok, gpt2_model, device),
                'fk_grade': compute_fk_grade(text),
                'persona_alignment': align,
                'kl_div': kl
            }
        
        # Record original
        results['original'].append(record(ai_text))
        
        # Run methods
        our_txt, _ = methods.our_method(ai_text, question, pool_df, persona_profile,
                                       det_tok, det_model, gpt2_tok, gpt2_model, 
                                       device=device, n_candidates=3)
        sico_txt, _ = methods.sico_baseline(ai_text, question, pool_df,
                                           det_tok, det_model, gpt2_tok, gpt2_model,
                                           device=device)
        sda_txt, _ = methods.sda_baseline(ai_text, question, pool_df,
                                         det_tok, det_model, gpt2_tok, gpt2_model,
                                         device=device)
        
        results['ours'].append(record(our_txt, original=ai_text))
        results['sico'].append(record(sico_txt, original=ai_text))
        results['sda'].append(record(sda_txt, original=ai_text))
    
    print("Evaluation complete!")
    return results


def compute_bertscore_for_methods(results, methods_list=['ours', 'sico', 'sda']):
    """
    Compute BERTScore for all methods and add to results dicts.
    """
    print("\nComputing BERTScores...")
    for method in methods_list:
        cands = [r['text'] for r in results[method]]
        refs = [r['original'] for r in results[method]]
        bs = compute_bertscore(cands, refs, device='cpu')
        for r in results[method]:
            r['bertscore'] = bs
        print(f"  {method.upper():6s} BERTScore F1: {bs:.4f}")


def summarize_results(results):
    """
    Create a summary table of all evaluation metrics.
    
    Returns:
        DataFrame with one row per method
    """
    summary = {}
    for method, items in results.items():
        if items:
            summary[method] = {
                'Evasion Rate (↑)': f"{np.mean([r['evaded'] for r in items]):.2%}",
                'Avg AI Score (↓)': f"{np.mean([r['ai_score'] for r in items]):.4f}",
                'BERTScore F1 (↑)': f"{np.mean([r.get('bertscore', 0) for r in items]):.4f}",
                'Perplexity (→)': f"{np.mean([r['ppl'] for r in items]):.1f}",
                'FK Grade (↓)': f"{np.mean([r['fk_grade'] for r in items]):.2f}",
                'Persona Align (↑)': f"{np.mean([r['persona_alignment'] for r in items]):.4f}",
                'KL-Div (↓)': f"{np.mean([r['kl_div'] for r in items]):.4f}",
            }
    
    return pd.DataFrame(summary).T


def compute_auc_scores(results, human_df, det_tok, det_model, device, n_samples=None):
    """
    Compute AUROC for each method vs human baseline.
    
    Args:
        results: Results dict from evaluation
        human_df: DataFrame of human-written texts
        det_tok: Detector tokenizer
        det_model: Detector model
        device: Device to run on
        n_samples: Number of human samples to compare (default: len(results['original']))
    
    Returns:
        dict of method -> AUROC scores
    """
    n_eval = len(results['original'])
    if n_samples is None:
        n_samples = n_eval
    
    print("\n--- Computing AUROC Scores ---")
    human_scores = [get_ai_score(t, det_tok, det_model, device)
                   for t in tqdm(human_df.head(n_samples)['answer'], desc='Human scores')]
    
    auc_scores = {}
    for method in ['original', 'ours', 'sico', 'sda']:
        ai_scores = [r['ai_score'] for r in results[method]]
        y_true = [0] * len(human_scores) + [1] * len(ai_scores)
        y_scores = human_scores + ai_scores
        auc = roc_auc_score(y_true, y_scores)
        auc_scores[method] = auc
        print(f"  {method.upper():10s}: AUROC = {auc:.4f}")
    
    return auc_scores
