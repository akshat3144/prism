"""Dataset loading and preparation."""

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from .. import config


def load_hc3_dataset(split=config.HC3_SPLIT):
    """
    Load HC3 (Human-ChatGPT Comparison) dataset from HuggingFace.
    
    Args:
        split: which split to load ('train', 'test', 'all')
    
    Returns:
        Pandas DataFrame with columns: question, answer, label (0=human, 1=AI)
    """
    print(f"Loading HC3 dataset (split='{split}')...")
    ds = load_dataset(config.HC3_DATASET, config.HC3_SPLIT, split='train')
    raw_df = ds.to_pandas()
    print(f"Loaded {len(raw_df)} raw entries")
    
    rows = []
    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc='Processing HC3'):
        q = row['question']
        
        # Human answers
        for ans in row['human_answers']:
            if ans and len(ans.strip()) > config.MIN_TEXT_LENGTH:
                rows.append({'question': q, 'answer': ans.strip(), 'label': 0})
        
        # ChatGPT answers
        for ans in row['chatgpt_answers']:
            if ans and len(ans.strip()) > config.MIN_TEXT_LENGTH:
                rows.append({'question': q, 'answer': ans.strip(), 'label': 1})
    
    df = pd.DataFrame(rows)
    return df


def prepare_train_eval_split(df, n_samples=config.N_SAMPLES_PER_CLASS, seed=config.SEED):
    """
    Split HC3 data into human and AI samples for training/evaluation.
    
    Args:
        df: Full HC3 DataFrame
        n_samples: Number of samples per class to extract
        seed: Random seed for reproducibility
    
    Returns:
        (human_df, ai_df) - DataFrames of human and AI answers respectively
    """
    human_df = df[df.label == 0].sample(n_samples, random_state=seed).reset_index(drop=True)
    ai_df = df[df.label == 1].sample(n_samples, random_state=seed).reset_index(drop=True)
    
    print(f"Prepared {len(human_df)} human + {len(ai_df)} AI samples")
    return human_df, ai_df


def load_and_prepare(n_samples=config.N_SAMPLES_PER_CLASS):
    """
    Full pipeline: load HC3, extract samples, and return ready-to-use DataFrames.
    """
    df = load_hc3_dataset()
    human_df, ai_df = prepare_train_eval_split(df, n_samples=n_samples)
    return human_df, ai_df
