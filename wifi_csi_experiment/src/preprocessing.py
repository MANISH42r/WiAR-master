import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d

class CSIPreprocessor:
    """
    Preprocessing pipeline for Wi-Fi CSI amplitude streams.
    """
    def __init__(self, 
                 target_sequence_length=500, 
                 normalization="minmax", 
                 remove_outliers=True, 
                 filter_type="median", 
                 median_kernel_size=5):
        self.target_sequence_length = target_sequence_length
        self.normalization = normalization
        self.remove_outliers = remove_outliers
        self.filter_type = filter_type
        self.median_kernel_size = median_kernel_size

    def preprocess_amplitude(self, amplitude_raw):
        """
        Takes raw amplitude array of shape (Npackets, Nsub=30, Nrx=3, Ntx=2)
        and converts it into a clean, normalized, fixed-length feature matrix.
        
        Output shape: (channels=180, target_sequence_length=500)
        """
        n_packets, n_sub, n_rx, n_tx = amplitude_raw.shape
        
        # 1. Flatten subcarrier/antenna dimensions: (Npackets, 180)
        flattened = amplitude_raw.reshape(n_packets, n_sub * n_rx * n_tx)

        # 2. Outlier / Noise Filtering per channel
        if self.filter_type == "median":
            filtered = np.zeros_like(flattened)
            k_size = self.median_kernel_size
            if k_size % 2 == 0:
                k_size += 1
            for ch in range(flattened.shape[1]):
                filtered[:, ch] = medfilt(flattened[:, ch], kernel_size=k_size)
        else:
            filtered = flattened.copy()

        # 3. Normalization (Min-Max per channel or global)
        if self.normalization == "minmax":
            min_val = filtered.min(axis=0, keepdims=True)
            max_val = filtered.max(axis=0, keepdims=True)
            denom = np.where((max_val - min_val) == 0, 1e-8, (max_val - min_val))
            normalized = (filtered - min_val) / denom
        elif self.normalization == "zscore":
            mean_val = filtered.mean(axis=0, keepdims=True)
            std_val = filtered.std(axis=0, keepdims=True)
            denom = np.where(std_val == 0, 1e-8, std_val)
            normalized = (filtered - mean_val) / denom
        else:
            normalized = filtered

        # 4. Fixed Temporal Interpolation / Resampling to target_sequence_length
        current_len = normalized.shape[0]
        if current_len == self.target_sequence_length:
            resampled = normalized
        else:
            x_old = np.linspace(0, 1, current_len)
            x_new = np.linspace(0, 1, self.target_sequence_length)
            f_interp = interp1d(x_old, normalized, axis=0, kind='linear', fill_value="extrapolate")
            resampled = f_interp(x_new)

        # Transpose to (channels=180, time=500) format standard for PyTorch 1D Conv
        processed = resampled.T.astype(np.float32)

        # Check for NaN / Inf safety
        if np.isnan(processed).any() or np.isinf(processed).any():
            processed = np.nan_to_num(processed, nan=0.0, posinf=1.0, neginf=0.0)

        return processed

if __name__ == '__main__':
    # Test preprocessor on synthetic CSI sample
    print("Testing CSIPreprocessor...")
    dummy_raw = np.random.uniform(0.5, 25.0, size=(300, 30, 3, 2))
    preprocessor = CSIPreprocessor(target_sequence_length=500, normalization="minmax")
    out = preprocessor.preprocess_amplitude(dummy_raw)
    print("Preprocessed Output Shape (Channels, SequenceLength):", out.shape)
    print("Data type:", out.dtype)
    print("Range: min =", out.min(), ", max =", out.max())
    print("Contains NaN:", np.isnan(out).any())
    print("Contains Inf:", np.isinf(out).any())
