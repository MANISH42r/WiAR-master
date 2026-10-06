import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM Model treating Wi-Fi CSI as a temporal sequence of subcarrier channels.
    Input shape: (batch_size, in_channels=180, sequence_length=500)
    """
    def __init__(self, input_size=180, hidden_size=128, num_layers=2, num_classes=16, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
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
        # Transpose to (batch_size, sequence_length=500, input_size=180) for PyTorch LSTM
        x = x.transpose(1, 2)
        
        # lstm_out shape: (batch_size, 500, hidden_size=128)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Mean pooling over temporal dimension for robust sequence representation
        feat = torch.mean(lstm_out, dim=1)
        
        out = self.fc(feat)
        return out

if __name__ == '__main__':
    model = LSTMModel(input_size=180, num_classes=16)
    dummy_input = torch.randn(16, 180, 500)
    out = model(dummy_input)
    print("LSTM Forward Pass Successful:")
    print("  Input shape :", dummy_input.shape)
    print("  Output shape:", out.shape)
