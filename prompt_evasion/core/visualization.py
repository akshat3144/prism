"""Visualization utilities for results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import config


def plot_lambda_sweep(lambda_df):
    """
    Plot lambda sweep results (evasion, alignment, BERTScore).
    
    Args:
        lambda_df: DataFrame with columns: lambda, evasion_rate, persona_alignment, bertscore
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Lambda (λ) Sweep: Evasion vs Quality Trade-off', fontsize=13, fontweight='bold')
    
    lams = lambda_df['lambda'].tolist()
    specs = [
        ('evasion_rate', 'Evasion Rate', 'bo-'),
        ('persona_alignment', 'Persona Alignment', 'rs-'),
        ('bertscore', 'BERTScore F1', 'g^-'),
    ]
    
    for ax, (col, ylabel, style) in zip(axes, specs):
        ax.plot(lams, lambda_df[col], style, linewidth=2, markersize=8)
        ax.set_xlabel('λ', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(lams)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_ablation_study(results):
    """
    Plot ablation study: compare 4 methods across 3 key metrics.
    
    Args:
        results: dict with method -> [list of result records]
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Ablation Study: Ours vs SICO vs SDA vs Original', fontsize=13, fontweight='bold')
    
    methods = ['original', 'ours', 'sico', 'sda']
    labels = ['Original AI', 'Ours', 'SICO', 'SDA']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    
    metric_specs = [
        ('Evasion Rate', lambda items: np.mean([r['evaded'] for r in items])),
        ('Persona Alignment', lambda items: np.mean([r['persona_alignment'] for r in items])),
        ('BERTScore F1', lambda items: np.mean([r.get('bertscore', 0) for r in items])),
    ]
    
    for ax, (name, fn) in zip(axes, metric_specs):
        vals = [fn(results[m]) for m in methods]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for b, v in zip(bars, vals):
            height = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, height + 0.01,
                   f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_comparison_boxplot(results, metric='ai_score'):
    """
    Create boxplots comparing distributions of a metric across methods.
    
    Args:
        results: dict with method -> [list of result records]
        metric: which metric to plot ('ai_score', 'ppl', 'fk_grade', etc.)
    """
    data = [results[m] for m in ['original', 'ours', 'sico', 'sda']]
    values = [[r[metric] for r in method_results] for method_results in data]
    labels = ['Original', 'Ours', 'SICO', 'SDA']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(values, labels=labels, patch_artist=True)
    
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'Distribution of {metric} Across Methods', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_all_figures(lambda_df, results, output_dir='results/figures'):
    """
    Save all generated figures to disk.
    
    Args:
        lambda_df: Lambda sweep results DataFrame
        results: Evaluation results dict
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig1 = plot_lambda_sweep(lambda_df)
    fig1.savefig(f'{output_dir}/lambda_sweep.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_ablation_study(results)
    fig2.savefig(f'{output_dir}/ablation_study.png', dpi=150, bbox_inches='tight')
    
    fig3 = plot_comparison_boxplot(results, metric='ai_score')
    fig3.savefig(f'{output_dir}/ai_score_distribution.png', dpi=150, bbox_inches='tight')
    
    print(f"Saved figures to {output_dir}/")
