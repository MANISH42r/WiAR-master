import os
import sys
import re
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from csi_parser import CSIParser
from preprocessing import CSIPreprocessor

def build_processed_dataset(config_path, samples_per_activity=6, max_subjects=10):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_dir = config['dataset']['raw_data_dir']
    processed_dir = config['dataset']['processed_data_dir']
    os.makedirs(processed_dir, exist_ok=True)

    activity_names = config['dataset']['activities']
    prep_cfg = config['preprocessing']

    preprocessor = CSIPreprocessor(
        target_sequence_length=prep_cfg['target_sequence_length'],
        normalization=prep_cfg['normalization'],
        remove_outliers=prep_cfg['remove_outliers'],
        filter_type=prep_cfg['filter_type'],
        median_kernel_size=prep_cfg['median_kernel_size']
    )
    parser = CSIParser()

    print("=" * 70)
    print("BUILDING PROCESSED CSI DATASET SUBSET")
    print("=" * 70)
    print(f"Raw data directory     : {raw_dir}")
    print(f"Processed output dir   : {processed_dir}")
    print(f"Samples / act / subject: {samples_per_activity}")

    # Map directories to subjects
    dir_entries = sorted(os.listdir(raw_dir))
    subject_file_map = {}

    for d in dir_entries:
        dir_path = os.path.join(raw_dir, d)
        if not os.path.isdir(dir_path):
            continue
        
        name = d.lower()
        if 'volunteer_7' in name: subj = 'volunteer_7'
        elif 'volunteer_8' in name: subj = 'volunteer_8'
        elif 'volunteer_9' in name: subj = 'volunteer_9'
        elif 'volunteer_10' in name: subj = 'volunteer_10'
        elif 'volunteer_a' in name: subj = 'volunteer_a'
        elif 'volunteer_b' in name: subj = 'volunteer_b'
        elif 'volunteer_c' in name: subj = 'volunteer_c'
        elif 'volunteerd' in name: subj = 'volunteer_d'
        elif 'volunteere' in name: subj = 'volunteer_e'
        elif 'volunteer_f' in name: subj = 'volunteer_f'
        else: subj = d

        if subj not in subject_file_map:
            subject_file_map[subj] = []

        for root, subdirs, filenames in os.walk(dir_path):
            for f in filenames:
                if f.endswith('.txt') or f.endswith('.m'):
                    continue
                subject_file_map[subj].append(os.path.join(root, f))

    metadata_rows = []
    sample_counter = 0

    for subj in sorted(list(subject_file_map.keys()))[:max_subjects]:
        files = subject_file_map[subj]
        
        # Group files by activity
        act_files = {}
        for fpath in files:
            fname = os.path.basename(fpath)
            m = re.search(r'a(\d+)_(\d+)', fname)
            if m:
                act_id = int(m.group(1))
                trial_id = int(m.group(2))
            else:
                m_act = re.search(r'a(\d+)', fname)
                if not m_act:
                    continue
                act_id = int(m_act.group(1))
                trial_id = 1

            if act_id not in act_files:
                act_files[act_id] = []
            act_files[act_id].append((trial_id, fpath))

        # Select up to samples_per_activity for each activity
        for act_id in sorted(list(act_files.keys())):
            selected = sorted(act_files[act_id], key=lambda x: x[0])[:samples_per_activity]
            
            for trial_id, fpath in selected:
                try:
                    parsed = parser.parse_file(fpath)
                    processed = preprocessor.preprocess_amplitude(parsed['amplitude'])
                    
                    sample_name = f"sample_{sample_counter:04d}_{subj}_a{act_id}_t{trial_id}.npy"
                    save_path = os.path.join(processed_dir, sample_name)
                    np.save(save_path, processed)

                    metadata_rows.append({
                        'sample_id': f"sample_{sample_counter:04d}",
                        'file_path': fpath,
                        'subject_id': subj,
                        'activity_id': act_id,
                        'activity_name': activity_names.get(act_id, f"activity_{act_id}"),
                        'trial_id': trial_id,
                        'processed_file': sample_name
                    })
                    sample_counter += 1
                except Exception as e:
                    print(f"Skipping corrupt file {fpath}: {str(e)}")

    metadata_df = pd.DataFrame(metadata_rows)
    meta_csv_path = os.path.join(processed_dir, 'metadata.csv')
    metadata_df.to_csv(meta_csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("PROCESSED DATASET GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total processed samples: {len(metadata_df)}")
    print(f"Subjects included      : {metadata_df['subject_id'].nunique()} ({list(metadata_df['subject_id'].unique())})")
    print(f"Activities included    : {metadata_df['activity_id'].nunique()}")
    print(f"Saved metadata to      : {meta_csv_path}")

    return metadata_df

if __name__ == '__main__':
    config_path = r"c:\Users\manishkumar.COMPILERX\Downloads\WiAR-master\wifi_csi_experiment\configs\config.yaml"
    build_processed_dataset(config_path, samples_per_activity=6, max_subjects=10)
