import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    Lightweight 1D Convolutional Neural Network for Wi-Fi CSI Activity Recognition.
    Input shape: (batch_size, in_channels=180, sequence_length=500)
    """
    def __init__(self, in_channels=180, num_classes=16, dropout=0.3):
        super(CNN1D, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 180, 500)
        feat = self.features(x)
        out = self.classifier(feat)
        return out

if __name__ == '__main__':
    model = CNN1D(in_channels=180, num_classes=16)
    dummy_input = torch.randn(16, 180, 500)
    out = model(dummy_input)
    print("CNN1D Forward Pass Successful:")
    print("  Input shape :", dummy_input.shape)
    print("  Output shape:", out.shape)
