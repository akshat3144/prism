"""Prompt construction for different methods."""

from .. import config


def build_persona_prompt(ai_text, question, pool_df, persona_profile, k=config.K_SHOTS, seed=config.SEED):
    """
    Build a persona-augmented prompt for humanization.
    
    Args:
        ai_text: AI-generated text to rewrite
        question: Original question
        pool_df: DataFrame with persona-aligned pool examples
        persona_profile: dict with stylometric persona averages
        k: Number of few-shot examples
        seed: Random seed
    
    Returns:
        String prompt for GPT-2 generation
    """
    examples = pool_df.sample(min(k, len(pool_df)), random_state=seed)
    
    desc = (
        f"Write like a human who uses contractions {persona_profile['contraction_rate']:.1%} of the time, "
        f"hedges with words like 'maybe'/'probably' ({persona_profile['hedging_density']:.1%} rate), "
        f"varies sentence lengths, and keeps a casual conversational tone."
    )
    
    prompt = (
        f"You are rewriting AI-generated text to sound more human.\n"
        f"Persona: {desc}\n"
        f"Preserve all original meaning. Do not add new facts.\n\n"
        f"Human-written examples:\n"
    )
    
    for _, ex in examples.iterrows():
        prompt += f"Q: {ex['question'][:100]}\nA: {ex['answer'][:250]}\n\n"
    
    prompt += (
        f"Now rewrite this AI answer in the human style above:\n"
        f"Q: {question[:150]}\n"
        f"AI Answer: {ai_text[:400]}\n"
        f"Human Rewrite:"
    )
    
    return prompt


def build_sico_prompt(ai_text, question, pool_df, k=3, seed=config.SEED):
    """
    Build SICO baseline prompt (no persona modeling, pure adversarial).
    
    Args:
        ai_text: AI-generated text
        question: Original question
        pool_df: DataFrame with examples
        k: Number of examples
        seed: Random seed
    
    Returns:
        String prompt for GPT-2 generation
    """
    examples = pool_df.sample(min(k, len(pool_df)), random_state=seed)
    
    prompt = "Rewrite the following text to sound more natural:\n\n"
    for _, ex in examples.iterrows():
        prompt += f"Example: {ex['answer'][:200]}\n\n"
    prompt += f"Rewrite: {ai_text[:400]}\nResult:"
    
    return prompt


def build_sda_prompt(ai_text, question, pool_df, k=5, seed=config.SEED):
    """
    Build SDA baseline prompt (retrieval-based context, no stylometric modeling).
    
    Args:
        ai_text: AI-generated text
        question: Original question
        pool_df: DataFrame with context examples
        k: Number of context examples
        seed: Random seed
    
    Returns:
        String prompt for GPT-2 generation
    """
    examples = pool_df.sample(min(k, len(pool_df)), random_state=seed + 1)
    
    prompt = f"Question: {question[:100]}\nHuman answers for context:\n"
    for _, ex in examples.iterrows():
        prompt += f"- {ex['answer'][:200]}\n"
    prompt += f"\nRewrite this to be more human-like:\n{ai_text[:400]}\nResponse:"
    
    return prompt
