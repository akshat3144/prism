"""Text generation utilities."""

import torch
from .. import config


def generate_rewrite(prompt, gpt2_tok, gpt2_model, device=config.DEVICE,
                     temperature=config.TEMPERATURE_DEFAULT,
                     top_p=config.TOP_P_DEFAULT,
                     max_new_tokens=config.MAX_NEW_TOKENS):
    """
    Generate a rewrite using GPT-2 from a given prompt.
    
    Args:
        prompt: Input prompt for generation
        gpt2_tok: GPT-2 tokenizer
        gpt2_model: GPT-2 model
        device: Device to run on
        temperature: Sampling temperature (higher = more diverse)
        top_p: Top-p (nucleus) sampling parameter
        max_new_tokens: Maximum new tokens to generate
    
    Returns:
        Generated text (or None if too short or generation fails)
    """
    inputs = gpt2_tok(prompt, return_tensors='pt', truncation=True, max_length=768)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        out = gpt2_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=config.REPETITION_PENALTY,
            pad_token_id=gpt2_tok.eos_token_id,
            eos_token_id=gpt2_tok.eos_token_id
        )
    
    # Decode only the newly generated part
    text = gpt2_tok.decode(out[0][input_len:], skip_special_tokens=True)
    text = text.strip().split('\n')[0].strip()
    
    # Return only if sufficiently long
    return text if len(text) > 30 else None


def generate_candidates(prompt, gpt2_tok, gpt2_model, temperatures, device=config.DEVICE):
    """
    Generate multiple candidates at different temperatures.
    
    Args:
        prompt: Input prompt
        gpt2_tok: GPT-2 tokenizer
        gpt2_model: GPT-2 model
        temperatures: List of temperature values to try
        device: Device to run on
    
    Returns:
        List of generated texts (non-None only)
    """
    candidates = []
    for temp in temperatures:
        text = generate_rewrite(prompt, gpt2_tok, gpt2_model, device=device, temperature=temp)
        if text:
            candidates.append(text)
    return candidates
