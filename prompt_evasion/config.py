"""Configuration and constants for Persona-Augmented Prompt Optimizer."""

import torch
import random
import numpy as np

# ============ SEED & DEVICE ============
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============ MODEL PATHS ============
DETECTOR_MODEL = 'roberta-base-openai-detector'
GPT2_MODEL = 'gpt2'
BERTSCORE_MODEL = 'distilbert-base-uncased'

# ============ DATASET CONFIG ============
HC3_DATASET = 'Hello-SimpleAI/HC3'
HC3_SPLIT = 'all'
MIN_TEXT_LENGTH = 50  # Minimum text length for filtering

# Sample sizes
N_SAMPLES_PER_CLASS = 500  # 500 human + 500 AI
N_POOL = 200  # Size of high-alignment persona pool
K_SHOTS = 3   # Few-shot examples in prompts
N_EVAL = 50   # Evaluation set size
N_SWEEP = 20  # Lambda sweep evaluation size

# ============ THRESHOLDS & PARAMETERS ============
THRESHOLD = 0.5  # AI score threshold for classification

# ============ GENERATION PARAMETERS ============
MAX_NEW_TOKENS = 180
TEMPERATURE_DEFAULT = 0.85
TOP_P_DEFAULT = 0.92
REPETITION_PENALTY = 1.2

# Temperature sweep for lambda tuning
TEMPERATURE_SWEEP = [0.7, 0.8, 0.9, 1.0, 1.1]
LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

# ============ LOSS WEIGHTS ============
# Default weights for combined loss: L = λ·L_adv + β·L_sem + γ·L_persona
LAMBDA_DEFAULT = 0.5    # Adversarial loss weight
BETA_DEFAULT = 0.3      # Semantic loss weight
GAMMA_DEFAULT = 0.2     # Persona loss weight

# ============ OUTPUT PATHS ============
OUTPUT_DIR = 'results'
RESULTS_DIR = 'results/method_outputs'
FIGURES_DIR = 'results/figures'
CSVS_DIR = 'results/csvs'

# ============ LOGGING ============
VERBOSE = True
