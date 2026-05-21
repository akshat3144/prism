"""Main entry point: full pipeline from data loading to evaluation."""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import config
from core import data, models, generation, prompts, evaluation, visualization, methods
from utils.stylometrics import build_persona_profile, extract_stylometrics, persona_alignment_score
from utils.metrics import get_ai_score, compute_perplexity, compute_bertscore


def main():
    """Full pipeline: load data, build persona, run evaluation, visualize results."""
    
    print("=" * 70)
    print("PRISM: Persona-Augmented Prompt Optimizer for AI Text Humanization")
    print("=" * 70)
    
    # ========== SETUP ==========
    print("\n[1] Loading models...")
    det_tok, det_model = models.load_detector_model(device=config.DEVICE)
    gpt2_tok, gpt2_model = models.load_gpt2_model(device=config.DEVICE)
    models.print_device_info(config.DEVICE)
    
    # ========== DATA LOADING ==========
    print("\n[2] Loading HC3 dataset...")
    human_df, ai_df = data.load_and_prepare(n_samples=config.N_SAMPLES_PER_CLASS)
    
    # ========== PERSONA EXTRACTION ==========
    print(f"\n[3] Building persona profile from {len(human_df)} human samples...")
    persona_profile = build_persona_profile(human_df['answer'].tolist())
    print("Human Persona Profile:")
    for k, v in persona_profile.items():
        print(f"  {k:25s}: {v:.4f}")
    
    # ========== PERSONA-ALIGNED POOL ==========
    print(f"\n[4] Building persona-aligned pool ({config.N_POOL} samples)...")
    
    pool_rows = []
    for _, row in tqdm(human_df.iterrows(), total=len(human_df), desc='Scoring'):
        feats = extract_stylometrics(row['answer'])
        alignment, kl = persona_alignment_score(feats, persona_profile)
        pool_rows.append({**row.to_dict(), 'alignment': alignment, 'kl_div': kl})
    
    pool_df = (pd.DataFrame(pool_rows)
               .sort_values('alignment', ascending=False)
               .head(config.N_POOL)
               .reset_index(drop=True))
    
    print(f"Pool built. Mean alignment: {pool_df['alignment'].mean():.4f}")
    
    # ========== MAIN EVALUATION ==========
    print(f"\n[5] Running evaluation on {config.N_EVAL} samples...")
    results = evaluation.run_evaluation(ai_df, pool_df, persona_profile,
                                       det_tok, det_model, gpt2_tok, gpt2_model,
                                       device=config.DEVICE, n_eval=config.N_EVAL)
    
    # ========== BERTSCORE COMPUTATION ==========
    evaluation.compute_bertscore_for_methods(results)
    
    # ========== SUMMARY STATISTICS ==========
    print("\n[6] Results Summary:")
    print("=" * 90)
    summary_df = evaluation.summarize_results(results)
    print(summary_df.to_string())
    print("=" * 90)
    
    # ========== AUROC SCORES ==========
    auc_scores = evaluation.compute_auc_scores(results, human_df, det_tok, det_model, config.DEVICE)
    
    # ========== LAMBDA SWEEP (Optional - takes time) ==========
    print(f"\n[7] Lambda sweep on {config.N_SWEEP} samples...")
    lambda_rows = []
    sweep_set = ai_df.head(config.N_SWEEP).reset_index(drop=True)
    
    for lam in config.LAMBDA_VALUES:
        evasions, alignments, rewrites, originals = [], [], [], []
        for _, row in tqdm(sweep_set.iterrows(), desc=f'λ={lam}', leave=False):
            txt, _ = methods.our_method(row['answer'], row['question'], pool_df,
                                       persona_profile, det_tok, det_model,
                                       gpt2_tok, gpt2_model, device=config.DEVICE,
                                       n_candidates=3, lam=lam)
            ai_s = get_ai_score(txt, det_tok, det_model, config.DEVICE)
            feats = extract_stylometrics(txt)
            align, _ = persona_alignment_score(feats, persona_profile)
            
            evasions.append(ai_s < config.THRESHOLD)
            alignments.append(align)
            rewrites.append(txt)
            originals.append(row['answer'])
        
        bs = compute_bertscore(rewrites, originals, device='cpu')
        lambda_rows.append({
            'lambda': lam,
            'evasion_rate': np.mean(evasions),
            'persona_alignment': np.mean(alignments),
            'bertscore': bs
        })
        print(f"  λ={lam} → Evasion={np.mean(evasions):.2%}  PersonaAlign={np.mean(alignments):.4f}  BERTScore={bs:.4f}")
    
    lambda_df = pd.DataFrame(lambda_rows)
    
    # ========== SAVE RESULTS ==========
    print("\n[8] Saving results...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CSVS_DIR, exist_ok=True)
    
    for method in ['original', 'ours', 'sico', 'sda']:
        pd.DataFrame(results[method]).to_csv(f'{config.CSVS_DIR}/results_{method}.csv', index=False)
    
    lambda_df.to_csv(f'{config.CSVS_DIR}/lambda_sweep.csv', index=False)
    summary_df.to_csv(f'{config.CSVS_DIR}/evaluation_summary.csv')
    
    print(f"Saved CSVs to {config.CSVS_DIR}/")
    
    # ========== VISUALIZATIONS ==========
    print("\n[9] Generating visualizations...")
    visualization.save_all_figures(lambda_df, results, output_dir=config.FIGURES_DIR)
    
    print("\n" + "=" * 70)
    print("Pipeline complete! Results saved to:")
    print(f"  CSVs:   {config.CSVS_DIR}/")
    print(f"  Plots:  {config.FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
