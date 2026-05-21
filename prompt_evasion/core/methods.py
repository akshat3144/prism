"""Method implementations: Ours, SICO baseline, SDA baseline."""

from .. import config
from . import prompts, generation, losses


def our_method(ai_text, question, pool_df, persona_profile, det_tok, det_model,
               gpt2_tok, gpt2_model, device=config.DEVICE, n_candidates=5,
               lam=config.LAMBDA_DEFAULT, beta=config.BETA_DEFAULT, gamma=config.GAMMA_DEFAULT):
    """
    Stylistic Persona-Augmented Prompt Optimizer.
    
    Generates multiple candidates with varied temperatures, picks the one with lowest combined loss.
    
    Args:
        ai_text: AI-generated text to humanize
        question: Original question
        pool_df: DataFrame with persona-aligned examples
        persona_profile: Target persona profile
        det_tok: Detector tokenizer
        det_model: Detector model
        gpt2_tok: GPT-2 tokenizer
        gpt2_model: GPT-2 model
        device: Device to run on
        n_candidates: Number of temperature samples
        lam, beta, gamma: Loss weights
    
    Returns:
        (best_text, best_metrics)
    """
    # Build prompt with persona guidance
    prompt = prompts.build_persona_prompt(ai_text, question, pool_df, persona_profile)
    temperatures = config.TEMPERATURE_SWEEP[:n_candidates]
    
    # Generate candidates
    candidates = []
    for temp in temperatures:
        cand = generation.generate_rewrite(prompt, gpt2_tok, gpt2_model, 
                                           device=device, temperature=temp)
        if cand:
            loss, metrics = losses.compute_combined_loss(
                ai_text, cand, det_tok, det_model, device, 
                persona_profile, lam, beta, gamma
            )
            candidates.append({'text': cand, 'loss': loss, 'metrics': metrics})
    
    if not candidates:
        return ai_text, {}
    
    best = min(candidates, key=lambda x: x['loss'])
    return best['text'], best['metrics']


def sico_baseline(ai_text, question, pool_df, det_tok, det_model,
                  gpt2_tok, gpt2_model, device=config.DEVICE, n_candidates=3):
    """
    SICO baseline: No persona, pure adversarial, static pool.
    
    Args:
        ai_text: AI-generated text
        question: Original question
        pool_df: DataFrame with examples
        det_tok: Detector tokenizer
        det_model: Detector model
        gpt2_tok: GPT-2 tokenizer
        gpt2_model: GPT-2 model
        device: Device to run on
        n_candidates: Number of temperature samples
    
    Returns:
        (best_text, best_metrics)
    """
    prompt = prompts.build_sico_prompt(ai_text, question, pool_df)
    best_text, best_score = ai_text, None
    
    for temp in [0.7, 0.9, 1.1][:n_candidates]:
        cand = generation.generate_rewrite(prompt, gpt2_tok, gpt2_model,
                                           device=device, temperature=temp)
        if cand:
            # Simple scoring: just minimize AI score
            inputs = det_tok(cand, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            import torch
            with torch.no_grad():
                logits = det_model(**inputs).logits
            score = torch.softmax(logits.float(), dim=-1)[0][1].item()
            
            if best_score is None or score < best_score:
                best_score, best_text = score, cand
    
    return best_text, {'ai_score': best_score if best_score is not None else 0}


def sda_baseline(ai_text, question, pool_df, det_tok, det_model,
                 gpt2_tok, gpt2_model, device=config.DEVICE, n_candidates=3):
    """
    SDA baseline: Retrieval-based context, no stylometric modeling.
    
    Args:
        ai_text: AI-generated text
        question: Original question
        pool_df: DataFrame with context examples
        det_tok: Detector tokenizer
        det_model: Detector model
        gpt2_tok: GPT-2 tokenizer
        gpt2_model: GPT-2 model
        device: Device to run on
        n_candidates: Number of temperature samples
    
    Returns:
        (best_text, best_metrics)
    """
    prompt = prompts.build_sda_prompt(ai_text, question, pool_df)
    best_text, best_score = ai_text, None
    
    for temp in [0.8, 0.95, 1.05][:n_candidates]:
        cand = generation.generate_rewrite(prompt, gpt2_tok, gpt2_model,
                                           device=device, temperature=temp)
        if cand:
            inputs = det_tok(cand, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            import torch
            with torch.no_grad():
                logits = det_model(**inputs).logits
            score = torch.softmax(logits.float(), dim=-1)[0][1].item()
            
            if best_score is None or score < best_score:
                best_score, best_text = score, cand
    
    return best_text, {'ai_score': best_score if best_score is not None else 0}
