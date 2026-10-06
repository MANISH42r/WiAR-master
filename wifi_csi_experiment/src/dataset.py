import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CSIDataset(Dataset):
    """
    PyTorch Dataset for Wi-Fi CSI Human Activity Recognition.
    """
    def __init__(self, metadata_df, processed_dir, transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing columns ['sample_id', 'subject_id', 'activity_id', 'processed_file']
            processed_dir (str): Base directory where preprocessed .npy files are cached.
            transform (callable, optional): Optional PyTorch transform/augmentation.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = row['processed_file']
        
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.processed_dir, file_path)

        # Load numpy array of shape (180, 500)
        data = np.load(file_path).astype(np.float32)
        
        if self.transform:
            data = self.transform(data)

        # Convert to PyTorch Tensor: shape (180, 500)
        tensor_data = torch.from_numpy(data)
        
        # Convert 1-indexed activity (a1-a16) to 0-indexed label (0-15)
        label = int(row['activity_id']) - 1
        tensor_label = torch.tensor(label, dtype=torch.long)

        return tensor_data, tensor_label

if __name__ == '__main__':
    print("Testing CSIDataset...")
    dummy_meta = pd.DataFrame([{
        'sample_id': 'sample_0',
        'subject_id': 'volunteer_7',
        'activity_id': 1,
        'processed_file': 'dummy.npy'
    }])
    # Save dummy file
    dummy_path = 'dummy.npy'
    np.save(dummy_path, np.zeros((180, 500), dtype=np.float32))
    
    dataset = CSIDataset(dummy_meta, '.')
    x, y = dataset[0]
    print(f"Sample tensor shape : {x.shape}")
    print(f"Sample label        : {y.item()} (activity 1 -> label 0)")
    
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
