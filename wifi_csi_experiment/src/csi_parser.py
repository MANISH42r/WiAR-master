import os
import numpy as np
import csiread

class CSIParser:
    """
    Parser for Intel 5300 802.11n Wi-Fi CSI binary (.dat) files using csiread.
    """
    def __init__(self, remove_zero_packets=True):
        self.remove_zero_packets = remove_zero_packets

    def parse_file(self, file_path):
        """
        Parses a single Intel CSI dat file.
        
        Returns:
            dict containing:
                - 'csi_complex': np.ndarray of shape (Npackets, Nsub=30, Nrx=3, Ntx=2)
                - 'amplitude': np.ndarray of shape (Npackets, Nsub=30, Nrx=3, Ntx=2)
                - 'phase': np.ndarray of shape (Npackets, Nsub=30, Nrx=3, Ntx=2)
                - 'n_packets': int
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSI file not found: {file_path}")

        try:
            csidata = csiread.Intel(file_path)
            csidata.read()
            csi_complex = csidata.get_scaled_csi()  # shape: (Npackets, 30, 3, 2)
        except Exception as e:
            raise RuntimeError(f"Error parsing CSI file {file_path}: {str(e)}")

        if csi_complex.size == 0 or csi_complex.shape[0] == 0:
            raise ValueError(f"No valid CSI packets found in file {file_path}")

        # Compute amplitude and phase
        amplitude = np.abs(csi_complex)
        phase = np.angle(csi_complex)

        # Remove dead/zero packets if requested
        if self.remove_zero_packets:
            valid_mask = amplitude.sum(axis=(1, 2, 3)) > 0
            if np.sum(valid_mask) > 0:
                csi_complex = csi_complex[valid_mask]
                amplitude = amplitude[valid_mask]
                phase = phase[valid_mask]

        return {
            'csi_complex': csi_complex,
            'amplitude': amplitude,
            'phase': phase,
            'n_packets': amplitude.shape[0]
        }

if __name__ == '__main__':
    # Diagnostic parser test on ONE sample file
    test_file = r"c:\Users\manishkumar.COMPILERX\Downloads\WiAR-master\WiAR-master\data\volunteer_7\volunteer_7\csi_a10_1.dat"
    print(f"Testing CSIParser on single file: {test_file}")
    parser = CSIParser()
    result = parser.parse_file(test_file)
    print("SUCCESSFULLY PARSED SINGLE CSI FILE:")
    print(f"  - Complex CSI shape : {result['csi_complex'].shape}")
    print(f"  - Amplitude shape   : {result['amplitude'].shape}")
    print(f"  - Phase shape       : {result['phase'].shape}")
    print(f"  - Packet Count      : {result['n_packets']}")
    print(f"  - Amplitude range   : min={result['amplitude'].min():.4f}, max={result['amplitude'].max():.4f}, mean={result['amplitude'].mean():.4f}")
