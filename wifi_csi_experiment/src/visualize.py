import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'font.family': 'sans-serif'
    })

def plot_random_vs_unseen(results_df, output_dir):
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    
    models = results_df['model'].unique()
    x = np.arange(len(models))
    width = 0.35

    random_accs = [results_df[(results_df['model']==m) & (results_df['experiment']=='random_split')]['accuracy'].values[0] for m in models]
    unseen_accs = [results_df[(results_df['model']==m) & (results_df['experiment']=='unseen_user')]['accuracy'].values[0] for m in models]

    rects1 = ax.bar(x - width/2, random_accs, width, label='Random Split (Seen Users)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, unseen_accs, width, label='Unseen-User Split (Generalization)', color='#d95f02')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cross-User Generalization: Random Split vs. Unseen-User Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 105)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'random_vs_unseen_accuracy.png'), dpi=300)
    plt.close()

def plot_performance_drop(results_df, output_dir):
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    models = results_df['model'].unique()
    abs_drops = []
    pct_drops = []

    for m in models:
        rand_acc = results_df[(results_df['model']==m) & (results_df['experiment']=='random_split')]['accuracy'].values[0]
        uns_acc = results_df[(results_df['model']==m) & (results_df['experiment']=='unseen_user')]['accuracy'].values[0]
        abs_drop = rand_acc - uns_acc
        pct_drop = ((rand_acc - uns_acc) / rand_acc) * 100.0 if rand_acc > 0 else 0.0
        abs_drops.append(abs_drop)
        pct_drops.append(pct_drop)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    x = np.arange(len(models))
    width = 0.35

    color1 = '#c72e29'
    color2 = '#7570b3'

    rects1 = ax1.bar(x - width/2, abs_drops, width, label='Absolute Accuracy Drop (%)', color=color1)
    ax1.set_ylabel('Absolute Accuracy Drop (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(abs_drops + [10]) * 1.3)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, pct_drops, width, label='Relative Performance Drop (%)', color=color2)
    ax2.set_ylabel('Relative Performance Drop (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(pct_drops + [10]) * 1.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title('Generalization Gap: Performance Drop When Testing on Unseen Users')

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'-{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color=color1, fontweight='bold')
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'-{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color=color2, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_drop_comparison.png'), dpi=300)
    plt.close()

def plot_training_curves(histories, experiment_name, output_dir):
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Plot Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'CNN': '#1b9e77', 'LSTM': '#d95f02', 'CNN-LSTM': '#7570b3'}

    for model_name, hist in histories.items():
        epochs = hist['epoch']
        c = colors.get(model_name, '#333333')
        ax1.plot(epochs, hist['train_loss'], label=f'{model_name} (Train)', linestyle='--', color=c, alpha=0.7)
        ax1.plot(epochs, hist['val_loss'], label=f'{model_name} (Val)', linestyle='-', color=c, linewidth=2)

        ax2.plot(epochs, hist['train_acc'], label=f'{model_name} (Train)', linestyle='--', color=c, alpha=0.7)
        ax2.plot(epochs, hist['val_acc'], label=f'{model_name} (Val)', linestyle='-', color=c, linewidth=2)

    ax1.set_title(f'Loss Curves ({experiment_name})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()

    ax2.set_title(f'Accuracy Curves ({experiment_name})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 105)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_{experiment_name}.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(cm, model_name, experiment_name, output_dir):
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [f"a{i}" for i in range(1, 17)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name} ({experiment_name})')
    ax.set_xlabel('Predicted Activity Class')
    ax.set_ylabel('True Activity Class')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_{experiment_name}_{model_name}.png'), dpi=300)
    plt.close()
