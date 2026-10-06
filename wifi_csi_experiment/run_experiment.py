import os
import sys
import argparse
import time
import yaml
import torch
import numpy as np
import pandas as pd

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from create_dataset import build_processed_dataset
from split_by_user import create_unseen_user_split, create_random_split
from dataset import CSIDataset
from models.cnn import CNN1D
from models.lstm import LSTMModel
from models.cnn_lstm import CNNLSTM
from train import train_model
from evaluate import evaluate_model
from visualize import (
    plot_random_vs_unseen,
    plot_performance_drop,
    plot_training_curves,
    plot_confusion_matrix
)

def get_model(model_name, config):
    m_cfg = config['models']
    name_clean = model_name.lower().replace('-', '_')
    if name_clean in ['cnn', 'cnn1d']:
        return CNN1D(in_channels=m_cfg['cnn']['in_channels'], 
                     num_classes=m_cfg['cnn']['num_classes'], 
                     dropout=m_cfg['cnn']['dropout'])
    elif name_clean == 'lstm':
        return LSTMModel(input_size=m_cfg['lstm']['input_size'],
                         hidden_size=m_cfg['lstm']['hidden_size'],
                         num_layers=m_cfg['lstm']['num_layers'],
                         num_classes=m_cfg['lstm']['num_classes'],
                         dropout=m_cfg['lstm']['dropout'])
    elif name_clean in ['cnn_lstm', 'cnnlstm']:
        return CNNLSTM(in_channels=m_cfg['cnn_lstm']['in_channels'],
                       conv_out_channels=m_cfg['cnn_lstm']['conv_out_channels'],
                       hidden_size=m_cfg['cnn_lstm']['hidden_size'],
                       num_layers=m_cfg['cnn_lstm']['num_layers'],
                       num_classes=m_cfg['cnn_lstm']['num_classes'],
                       dropout=m_cfg['cnn_lstm']['dropout'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Wi-Fi CSI HAR Cross-User Generalization Experiment")
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'config.yaml')
    parser.add_argument('--config', type=str, default=default_config, help='Path to config file')
    parser.add_argument('--experiment', type=str, default='all', choices=['all', 'random', 'unseen_user'], help='Experiment protocol')
    parser.add_argument('--model', type=str, default='all', choices=['all', 'cnn', 'lstm', 'cnn_lstm'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs')
    parser.add_argument('--samples_per_activity', type=int, default=6, help='Samples per activity per subject')
    parser.add_argument('--rebuild_dataset', action='store_true', help='Force rebuild preprocessed dataset')
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    seed = config['dataset'].get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print("RUNNING WI-FI CSI HAR CROSS-USER GENERALIZATION EXPERIMENTS")
    print(f"Device: {device} | Random Seed: {seed} | Config: {config_path}")
    print("=" * 80)

    # 1. Dataset Generation / Loading
    processed_dir = config['dataset']['processed_data_dir']
    meta_csv_path = os.path.join(processed_dir, 'metadata.csv')

    if args.rebuild_dataset or not os.path.exists(meta_csv_path):
        metadata_df = build_processed_dataset(config_path, samples_per_activity=args.samples_per_activity)
    else:
        print(f"Loading existing processed metadata from: {meta_csv_path}")
        metadata_df = pd.read_csv(meta_csv_path)

    print(f"Loaded Total Samples: {len(metadata_df)} across {metadata_df['subject_id'].nunique()} subjects.")

    # Determine experiments to run
    experiments_to_run = ['random_split', 'unseen_user'] if args.experiment == 'all' else [args.experiment]
    models_to_run = ['CNN', 'LSTM', 'CNN-LSTM'] if args.model == 'all' else [args.model.upper()]

    results_list = []
    histories_by_exp = {'random_split': {}, 'unseen_user': {}}

    results_dir = os.path.abspath('results')
    models_dir = os.path.join(results_dir, 'models')
    figures_dir = os.path.join(results_dir, 'figures')
    cm_dir = os.path.join(results_dir, 'confusion_matrices')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    for exp_name in experiments_to_run:
        print("\n" + "#" * 80)
        print(f"EXECUTING PROTOCOL: {exp_name.upper()}")
        print("#" * 80)

        if exp_name == 'random_split':
            train_df, test_df = create_random_split(metadata_df, test_ratio=0.2, random_seed=seed)
        else:
            test_subjects = config['split']['unseen_user']['test_subjects']
            train_df, test_df = create_unseen_user_split(metadata_df, test_subjects)

        train_dataset = CSIDataset(train_df, processed_dir)
        test_dataset = CSIDataset(test_df, processed_dir)

        for m_name in models_to_run:
            print("\n" + "-" * 70)
            print(f"Training Model: {m_name} | Experiment: {exp_name}")
            print("-" * 70)

            model = get_model(m_name, config)
            ckpt_path = os.path.join(models_dir, f"{exp_name}_{m_name}.pth")

            trained_model, history, train_time = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=test_dataset,  # Using test set as validation for clean logging
                config=config,
                checkpoint_path=ckpt_path,
                device=device
            )

            histories_by_exp[exp_name][m_name] = history

            # Evaluate Model
            metrics = evaluate_model(trained_model, test_dataset, device=device)

            print(f"Results for {m_name} ({exp_name}):")
            print(f"  - Test Accuracy : {metrics['accuracy']:.2f}%")
            print(f"  - Precision (Mac): {metrics['precision']:.2f}%")
            print(f"  - Recall (Mac)   : {metrics['recall']:.2f}%")
            print(f"  - F1-Score (Mac) : {metrics['f1_macro']:.2f}%")
            print(f"  - Training Time  : {train_time:.2f}s")
            print(f"  - Inference Time : {metrics['inference_time_sec']:.4f}s ({metrics['inference_time_ms_per_sample']:.2f} ms/sample)")

            # Save Confusion Matrix Plots to both figures and confusion_matrices directories
            plot_confusion_matrix(metrics['confusion_matrix'], m_name, exp_name, cm_dir)
            plot_confusion_matrix(metrics['confusion_matrix'], m_name, exp_name, figures_dir)

            results_list.append({
                'experiment': exp_name,
                'model': m_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'training_time': train_time,
                'inference_time': metrics['inference_time_sec']
            })

    # Save Results CSV
    results_df = pd.DataFrame(results_list)
    results_csv_path = os.path.join(results_dir, 'results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved numerical results to {results_csv_path}")

    # Generate Graphs if both experiments were executed
    if len(experiments_to_run) == 2 and len(models_to_run) >= 2:
        print("\n" + "=" * 80)
        print("GENERATING EXPERIMENTAL COMPARISON GRAPHS & VISUALIZATIONS")
        print("=" * 80)
        plot_random_vs_unseen(results_df, figures_dir)
        plot_performance_drop(results_df, figures_dir)
        plot_training_curves(histories_by_exp['random_split'], 'random_split', figures_dir)
        plot_training_curves(histories_by_exp['unseen_user'], 'unseen_user', figures_dir)
        print(f"Saved all figures to {figures_dir}")

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("EXPERIMENTAL SUMMARY TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))

    if len(experiments_to_run) == 2:
        print("\n" + "=" * 80)
        print("PERFORMANCE DROP ANALYSIS (RANDOM SPLIT vs. UNSEEN USER)")
        print("=" * 80)
        for m in models_to_run:
            rand_acc = results_df[(results_df['model']==m) & (results_df['experiment']=='random_split')]['accuracy'].values[0]
            uns_acc = results_df[(results_df['model']==m) & (results_df['experiment']=='unseen_user')]['accuracy'].values[0]
            abs_drop = rand_acc - uns_acc
            pct_drop = ((rand_acc - uns_acc) / rand_acc) * 100.0 if rand_acc > 0 else 0.0
            print(f"Model: {m:<10} | Random Acc: {rand_acc:6.2f}% | Unseen-User Acc: {uns_acc:6.2f}% | Absolute Drop: {abs_drop:6.2f}% | Pct Drop: {pct_drop:6.2f}%")

if __name__ == '__main__':
    main()
