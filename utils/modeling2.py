import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=(kernel_size - 1) // 2, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=stride, padding=(kernel_size - 1) // 2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.residual_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out
    
class TemporalResNet(nn.Module):
    """ResNet model with temporal convolutional layers."""
    def __init__(self, input_channels, num_classes, num_blocks, hidden_channels=64, kernel_size=3, dilation=1, dropout=0.2):
        super(TemporalResNet, self).__init__()
        self.input_layer = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)

        # Stack the specified number of residual blocks
        layers = []
        for _ in range(num_blocks):
            layers.append(TemporalConvBlock(hidden_channels, hidden_channels, kernel_size, dilation, dropout=dropout))
        self.residual_layers = nn.Sequential(*layers)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.global_avg_pool(x).squeeze(-1)  # Global average pooling across time dimension
        x = self.output_layer(x)
        return x
    
class TemporalResNetMultiInput(nn.Module):
    def __init__(self, imu_channels, video_channels, num_classes, num_blocks, hidden_channels=64, kernel_size=3, dilation=1, dropout=0.2):
        super(TemporalResNetMultiInput, self).__init__()

        # Separate input layers for IMU and video
        self.imu_input_layer = nn.Conv1d(imu_channels, hidden_channels, kernel_size=1)
        self.video_input_layer = nn.Conv1d(video_channels, hidden_channels, kernel_size=1)

        # Separate residual layers for IMU and video
        imu_layers = [TemporalConvBlock(hidden_channels, hidden_channels, kernel_size, dilation, dropout=dropout) for _ in range(num_blocks)]
        video_layers = [TemporalConvBlock(hidden_channels, hidden_channels, kernel_size, dilation, dropout=dropout) for _ in range(num_blocks)]
        self.imu_residual_layers = nn.Sequential(*imu_layers)
        self.video_residual_layers = nn.Sequential(*video_layers)

        # Global pooling to ensure compatibility for different input sequence lengths
        self.imu_global_pool = nn.AdaptiveAvgPool1d(1)
        self.video_global_pool = nn.AdaptiveAvgPool1d(1)

        # Output layer after fusion
        self.output_layer = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, imu_data, video_data):
        # Process IMU data
        imu_data = self.imu_input_layer(imu_data)
        imu_data = self.imu_residual_layers(imu_data)
        imu_data = self.imu_global_pool(imu_data).squeeze(-1)  # Global avg pooling to get (batch_size, hidden_channels)

        # Process video data
        video_data = self.video_input_layer(video_data)
        video_data = self.video_residual_layers(video_data)
        video_data = self.video_global_pool(video_data).squeeze(-1)  # Global avg pooling to get (batch_size, hidden_channels)

        # Concatenate pooled outputs
        combined = torch.cat([imu_data, video_data], dim=1)  # Shape: (batch_size, hidden_channels * 2)
        output = self.output_layer(combined)
        return output