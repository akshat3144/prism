"""Loss computation for humanization objectives."""

from .. import config
from ..utils.stylometrics import extract_stylometrics, persona_alignment_score


def compute_combined_loss(original, candidate, det_tok, det_model, device,
                          persona_profile, lam=config.LAMBDA_DEFAULT,
                          beta=config.BETA_DEFAULT, gamma=config.GAMMA_DEFAULT):
    """
    Compute combined loss: L = λ·L_adv + β·L_sem + γ·L_persona
    
    Args:
        original: Original AI-generated text
        candidate: Candidate humanized text
        det_tok: Detector tokenizer
        det_model: Detector model
        device: Device to run on
        persona_profile: Target persona profile dict
        lam: Adversarial loss weight
        beta: Semantic consistency weight
        gamma: Persona alignment weight
    
    Returns:
        (combined_loss, metrics_dict)
    """
    # Validation
    if not candidate or len(candidate.strip()) < 20:
        return float('inf'), {}
    
    # 1. L_adv: Adversarial loss (minimize P(AI))
    inputs = det_tok(candidate, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    import torch
    with torch.no_grad():
        logits = det_model(**inputs).logits
    ai_score = torch.softmax(logits.float(), dim=-1)[0][1].item()
    L_adv = ai_score
    
    # 2. L_sem: Semantic loss (preserve meaning via word overlap)
    orig_w = set(original.lower().split())
    cand_w = set(candidate.lower().split())
    L_sem = 1.0 - len(orig_w & cand_w) / max(len(orig_w), 1)
    
    # 3. L_persona: Persona alignment (KL-divergence)
    feats = extract_stylometrics(candidate)
    alignment, kl = persona_alignment_score(feats, persona_profile)
    L_persona = min(kl, 5.0)  # Clip KL to avoid large gradients
    
    # Combined loss
    combined = lam * L_adv + beta * L_sem + gamma * L_persona
    
    return combined, {
        'L_adv': L_adv,
        'L_sem': L_sem,
        'L_persona': L_persona,
        'combined': combined,
        'ai_score': ai_score,
        'alignment': alignment,
        'kl_div': kl
    }
