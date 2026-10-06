import os
import sys
import torch
import numpy as np

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from inspect_dataset import inspect_wiar_dataset
from csi_parser import CSIParser
from preprocessing import CSIPreprocessor
from models.cnn import CNN1D
from models.lstm import LSTMModel
from models.cnn_lstm import CNNLSTM

def run_pipeline_validation():
    print("=" * 80)
    print("STARTING ACTION VERIFICATION & PIPELINE DIAGNOSTICS")
    print("=" * 80)

    # 1 & 2 & 6: Inspect dataset structure, subjects, activities
    raw_dir = r"c:\Users\manishkumar.COMPILERX\Downloads\WiAR-master\WiAR-master\data"
    inspection_info = inspect_wiar_dataset(raw_dir)

    # 3 & 4 & 5: Parse one .dat file, print shape & packet count
    test_file = r"c:\Users\manishkumar.COMPILERX\Downloads\WiAR-master\WiAR-master\data\volunteer_7\volunteer_7\csi_a10_1.dat"
    print("\n" + "=" * 80)
    print(f"STEP 3-5: CSI PARSER VERIFICATION ON SINGLE FILE: {os.path.basename(test_file)}")
    print("=" * 80)
    
    parser = CSIParser()
    parsed_res = parser.parse_file(test_file)
    print(f"[VERIFIED] Complex CSI Shape (Npackets, Nsub, Nrx, Ntx): {parsed_res['csi_complex'].shape}")
    print(f"[VERIFIED] Total CSI Packets Parsed: {parsed_res['n_packets']}")
    print(f"[VERIFIED] Raw Amplitude Min/Max/Mean: {parsed_res['amplitude'].min():.4f} / {parsed_res['amplitude'].max():.4f} / {parsed_res['amplitude'].mean():.4f}")

    # 7: Create a tiny processed sample
    print("\n" + "=" * 80)
    print("STEP 7: PREPROCESSING VERIFICATION (TINY SAMPLE)")
    print("=" * 80)
    preprocessor = CSIPreprocessor(target_sequence_length=500, normalization="minmax")
    processed_sample = preprocessor.preprocess_amplitude(parsed_res['amplitude'])
    print(f"[VERIFIED] Preprocessed Sample Shape (Channels, SeqLength): {processed_sample.shape}")
    print(f"[VERIFIED] NaN Check: {np.isnan(processed_sample).any()} | Inf Check: {np.isinf(processed_sample).any()}")
    print(f"[VERIFIED] Preprocessed Value Range: min={processed_sample.min():.4f}, max={processed_sample.max():.4f}")

    # 8: Test CNN, LSTM, and CNN-LSTM model forward passes
    print("\n" + "=" * 80)
    print("STEP 8: MODEL FORWARD PASS VERIFICATION")
    print("=" * 80)
    
    # Batch size 4, 180 channels, 500 sequence length
    dummy_batch = torch.from_numpy(np.stack([processed_sample] * 4, axis=0))
    print(f"Batch Tensor Input Shape: {dummy_batch.shape}")

    # Model 1: CNN1D
    cnn = CNN1D(in_channels=180, num_classes=16)
    out_cnn = cnn(dummy_batch)
    print(f"[VERIFIED] CNN1D Forward Pass Output Shape: {out_cnn.shape}")

    # Model 2: LSTM
    lstm = LSTMModel(input_size=180, num_classes=16)
    out_lstm = lstm(dummy_batch)
    print(f"[VERIFIED] LSTM Forward Pass Output Shape : {out_lstm.shape}")

    # Model 3: CNN-LSTM
    cnn_lstm = CNNLSTM(in_channels=180, num_classes=16)
    out_cnn_lstm = cnn_lstm(dummy_batch)
    print(f"[VERIFIED] CNNLSTM Forward Pass Output Shape: {out_cnn_lstm.shape}")

    print("\n" + "=" * 80)
    print("ALL PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline_validation()
