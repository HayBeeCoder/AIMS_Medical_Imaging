"""
MedNet: Lightweight Attention-Augmented CNN for Medical Image Classification

Paper: https://www.nature.com/articles/s41598-025-25857-w
Authors: Md. Ferdous, Saifuddin Mahmud, Md. Eleush Zahan Shimul

This implementation follows the architecture described in the paper:
- Depthwise Separable Convolutions for parameter efficiency
- CBAM (Convolutional Block Attention Module) for spatial and channel attention
- ResidualDSCBAMBlock as the core building block
- 5 stages with progressive channel expansion (64 -> 128 -> 256 -> 512 -> 1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== CBAM Module (Based on Official Implementation) ==============

class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    Applies attention across channel dimension using both avg and max pooling.
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        reduced_channels = max(channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        
        # Max pooling path
        max_out = self.mlp(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    Applies attention across spatial dimensions using channel-wise pooling.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.bn(self.conv(combined)))
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially applies channel attention followed by spatial attention.
    
    Paper: https://arxiv.org/abs/1807.06521
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============== Depthwise Separable Convolution ==============

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.
    Consists of a depthwise convolution (3x3) followed by a pointwise convolution (1x1).
    This reduces parameters compared to standard convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Depthwise convolution (3x3, groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


# ============== ResidualDSCBAMBlock ==============

class ResidualDSCBAMBlock(nn.Module):
    """
    Core building block of MedNet.
    
    Structure:
    - Two consecutive depthwise separable convolutions
    - CBAM attention module
    - Residual connection (with 1x1 conv for dimension matching if needed)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Two consecutive depthwise separable convolutions
        self.dsc1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.dsc2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)
        
        # CBAM attention module
        self.cbam = CBAM(out_channels)
        
        # Residual connection (match dimensions if needed)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.dsc1(x)
        out = self.dsc2(out)
        out = self.cbam(out)
        
        out = out + residual
        return F.relu(out)


# ============== MedNet Architecture ==============

class MedNet(nn.Module):
    """
    MedNet: Lightweight Attention-Augmented CNN for Medical Image Classification.
    
    Paper: https://www.nature.com/articles/s41598-025-25857-w
    
    Architecture:
    - Stem: Initial 3x3 convolution
    - 5 stages of ResidualDSCBAMBlocks (64 -> 128 -> 256 -> 512 -> 1024 filters)
    - Adaptive average pooling
    - Dropout (default 0.4)
    - Two fully connected layers (1024 -> 256 -> num_classes)
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        num_classes (int): Number of output classes
        dropout (float): Dropout probability (default: 0.4)
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.4):
        super().__init__()
        
        # Initial convolution to expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 5 Stages of ResidualDSCBAMBlocks
        self.stage1 = ResidualDSCBAMBlock(64, 64, stride=1)
        self.stage2 = ResidualDSCBAMBlock(64, 128, stride=2)
        self.stage3 = ResidualDSCBAMBlock(128, 256, stride=2)
        self.stage4 = ResidualDSCBAMBlock(256, 512, stride=2)
        self.stage5 = ResidualDSCBAMBlock(512, 1024, stride=2)
        
        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        # Classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before the classifier (for visualization/analysis)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# ============== Focal Loss (for Class Imbalance) ==============

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal Loss down-weights easy examples and focuses on hard-to-classify samples.
    Particularly useful for imbalanced datasets like PneumoniaMNIST.
    
    Paper: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float): Weighting factor for the rare class (default: 1)
        gamma (float): Focusing parameter - higher values focus more on hard examples (default: 2)
        reduction (str): Specifies the reduction to apply: 'none', 'mean', 'sum'
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============== Utility Functions ==============

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_mednet_for_pneumonia(dropout=0.4):
    """
    Create MedNet configured for PneumoniaMNIST.
    
    Args:
        dropout (float): Dropout probability
    
    Returns:
        MedNet model configured for 1-channel grayscale input and 2-class output
    """
    return MedNet(
        in_channels=1,    # Grayscale chest X-rays
        num_classes=2,    # Normal vs Pneumonia
        dropout=dropout
    )


# ============== Test Code ==============

if __name__ == "__main__":
    # Test the model
    model = create_mednet_for_pneumonia()
    print(f"MedNet Parameters: {count_parameters(model):,}")
    
    # Test with PneumoniaMNIST input size (28x28)
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    # Test Focal Loss
    criterion = FocalLoss()
    targets = torch.tensor([0, 1, 1, 0])
    loss = criterion(output, targets)
    print(f"Focal Loss: {loss.item():.4f}")
