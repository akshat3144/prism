# Stylistic Persona-Augmented Prompt Optimizer

**Project Component:** Prompt-Level Evasion & Stylometric Optimization

**Format:** Modular Python Scripts (refactored from Jupyter Notebook)

This repository contains the full implementation of a Persona-Augmented Prompt Optimizer. The system is designed to transform AI-generated text into human-like prose by jointly optimizing for Adversarial Evasion, Semantic Consistency, and Stylometric Naturalness.

## Module Overview

Most AI text humanizers focus solely on breaking statistical patterns (like perplexity), often resulting in text that feels unnatural. This framework introduces a Persona-Augmented approach that:

- **Extracts Stylometric Fingerprints:** Profiles human writing patterns (e.g., contraction rates, hedging density, and sentence variance).
- **Optimizes via Hybrid Loss:** Employs a weighted objective function `L = λL_adv + βL_sem + γL_persona`.
- **Stress-Tests Detectors:** Specifically targets Neural classifiers (RoBERTa) to expose vulnerabilities in supervised detection models.

## Methodology & Loss Function

The optimizer selects the best rewrite candidate by minimizing a three-part loss:

- **Adversarial Loss (L_adv):** Minimizes the "AI-probability" score from the surrogate RoBERTa detector.
- **Semantic Consistency Loss (L_sem):** Uses BERTScore to ensure the humanized output preserves the original meaning.
- **Persona Alignment Loss (L_persona):** A novel constraint using KL-Divergence to align the output's style with a target human persona profile.

## 6-Metric Evaluation Protocol

The performance is validated using a rigorous evaluation suite implemented within the notebook:

| Metric        | Goal       | Description                                                  |
| ------------- | ---------- | ------------------------------------------------------------ |
| Evasion Rate  | ↑ High     | Percentage of samples that successfully bypass the detector. |
| Avg AI Score  | ↓ Low      | The raw probability score assigned by the AI detector.       |
| BERTScore F1  | ↑ High     | Semantic similarity score to ensure meaning preservation.    |
| Perplexity    | Balanced   | Fluency and predictability measured via GPT-2.               |
| FK Grade      | Human-like | Flesch-Kincaid grade level to match human readability.       |
| Persona Align | ↑ High     | Exponential negative KL-Div relative to the human profile.   |

## Implementation & Outputs

The entire project is now organized into modular Python scripts. Running the main pipeline will:

1. **Load Models & Data** — Downloads HC3 dataset, initializes RoBERTa detector and GPT-2 generator
2. **Build Persona Profile** — Extracts stylometric fingerprints from human-written texts
3. **Create Persona Pool** — Ranks and selects high-alignment examples for few-shot prompting
4. **Run Evaluation** — Tests all 4 methods (Original, Ours, SICO, SDA) on 50 samples
5. **Compute BERTScores** — Semantic similarity measurements across all methods
6. **Lambda Sweep** — Ablates the adversarial loss weight (λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0})
7. **Export Results** — Saves `results_*.csv`, `lambda_sweep.csv`, `evaluation_summary.csv`
8. **Visualize** — Generates trade-off plots and comparison charts

## Module Structure

```
prompt_evasion/
├── main.py                 # Entry point: full pipeline
├── config.py               # Configuration and constants
├── data.py                 # Dataset loading (HC3)
├── models.py               # Model initialization (RoBERTa, GPT-2)
├── generation.py           # Text generation utilities
├── prompts.py              # Prompt construction for each method
├── losses.py               # Combined loss computation
├── methods.py              # Method implementations (Ours, SICO, SDA)
├── evaluation.py           # Evaluation pipeline
├── visualization.py        # Plotting and result visualization
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── metrics.py          # Metric functions (AI score, perplexity, BERTScore, FK grade)
│   └── stylometrics.py     # Stylometric feature extraction & persona management
├── NLP_1.ipynb            # Supplementary experiments
└── README.md               # This file
```

## Quick Start

### Installation

```bash
cd ..  # Go to project root
pip install -r requirements.txt
cd prompt_evasion
```

### Run Full Pipeline

```bash
python main.py
```

This will:

- Load models and HC3 data
- Build persona profile from human texts
- Run evaluation across all 4 methods
- Perform lambda sweep
- Generate visualizations
- Save all results to `results/csvs/` and `results/figures/`

### Run Individual Components

```python
# Load data and models
import config, data, models

human_df, ai_df = data.load_and_prepare()
det_tok, det_model = models.load_detector_model(device=config.DEVICE)
gpt2_tok, gpt2_model = models.load_gpt2_model(device=config.DEVICE)

# Extract persona
from utils.stylometrics import build_persona_profile
persona_profile = build_persona_profile(human_df['answer'].tolist())

# Run a single method
import methods
rewritten_text, metrics = methods.our_method(
    ai_text="Your AI-generated text here",
    question="Original question",
    pool_df=pool_df,
    persona_profile=persona_profile,
    det_tok=det_tok, det_model=det_model,
    gpt2_tok=gpt2_tok, gpt2_model=gpt2_model,
    device=config.DEVICE
)
```

## Tech Stack

- **Models:** `roberta-base-openai-detector`, `gpt2`
- **Frameworks:** `transformers`, `torch`, `bert-score`, `textstat`
- **Dataset:** HC3 (Human-ChatGPT Comparison Corpus)
- **Evaluation:** `scikit-learn` (ROC-AUC), `scipy` (KL-divergence)
