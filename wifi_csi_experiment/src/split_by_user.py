import random
from collections import defaultdict

def create_unseen_user_split(metadata_df, test_subject_ids):
    """
    Creates an unseen-user split where specified test_subject_ids are strictly isolated in the test set.
    """
    all_subjects = sorted(list(metadata_df['subject_id'].unique()))
    test_subjects = set(test_subject_ids)
    train_subjects = set(all_subjects) - test_subjects

    overlap = train_subjects.intersection(test_subjects)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Subject leakage detected! Overlapping subjects: {overlap}")

    train_df = metadata_df[metadata_df['subject_id'].isin(train_subjects)].copy()
    test_df = metadata_df[metadata_df['subject_id'].isin(test_subjects)].copy()

    print("=" * 60)
    print("UNSEEN-USER SPLIT VERIFICATION")
    print("=" * 60)
    print(f"Training subjects ({len(train_subjects)}): {sorted(list(train_subjects))}")
    print(f"Testing subjects  ({len(test_subjects)}) : {sorted(list(test_subjects))}")
    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of testing samples : {len(test_df)}")
    print(f"Subject overlap: {len(overlap)}")
    print("Verification Passed: ZERO subject leakage.")
    print("=" * 60)

    return train_df, test_df

def create_random_split(metadata_df, test_ratio=0.2, random_seed=42):
    """
    Creates a conventional random train/test split (with potential subject overlap).
    """
    shuffled_df = metadata_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    num_test = int(len(shuffled_df) * test_ratio)
    
    test_df = shuffled_df.iloc[:num_test].copy()
    train_df = shuffled_df.iloc[num_test:].copy()

    train_subjs = set(train_df['subject_id'].unique())
    test_subjs = set(test_df['subject_id'].unique())

    print("=" * 60)
    print("RANDOM SPLIT SUMMARY (WARNING: SUBJECT LEAKAGE POSSIBLE)")
    print("=" * 60)
    print(f"Training subjects ({len(train_subjs)}): {sorted(list(train_subjs))}")
    print(f"Testing subjects  ({len(test_subjs)}) : {sorted(list(test_subjs))}")
    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of testing samples : {len(test_df)}")
    print(f"Subject overlap count     : {len(train_subjs.intersection(test_subjs))}")
    print("=" * 60)

    return train_df, test_df
