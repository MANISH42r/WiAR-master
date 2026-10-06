import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM Architecture for Wi-Fi CSI Activity Recognition.
    Combines 1D CNN feature extraction with LSTM temporal sequence modeling.
    Input shape: (batch_size, in_channels=180, sequence_length=500)
    """
    def __init__(self, in_channels=180, conv_out_channels=64, hidden_size=128, num_layers=1, num_classes=16, dropout=0.3):
        super(CNNLSTM, self).__init__()
        
        # CNN Feature Extractor along temporal sequence
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, conv_out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), # Reduces 500 -> 250 timesteps
            
            nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True)
        )
        
        # LSTM for temporal dynamics on extracted CNN feature sequences
        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 180, 500)
        cnn_feat = self.cnn(x)  # shape: (batch_size, conv_out_channels=64, 250)
        
        # Transpose to (batch_size, sequence_length=250, input_size=64)
        cnn_feat = cnn_feat.transpose(1, 2)
        
        lstm_out, _ = self.lstm(cnn_feat)  # shape: (batch_size, 250, hidden_size=128)
        
        # Global temporal pooling
        feat = torch.mean(lstm_out, dim=1)
        
        out = self.fc(feat)
        return out

if __name__ == '__main__':
    model = CNNLSTM(in_channels=180, num_classes=16)
    dummy_input = torch.randn(16, 180, 500)
    out = model(dummy_input)
    print("CNNLSTM Forward Pass Successful:")
    print("  Input shape :", dummy_input.shape)
    print("  Output shape:", out.shape)
