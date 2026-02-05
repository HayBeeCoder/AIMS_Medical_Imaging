"""
Data Preprocessing Pipeline for PneumoniaMNIST Binary Classification

This module provides comprehensive preprocessing utilities including:
- Normalization and augmentation transforms
- CLAHE contrast enhancement  
- Class imbalance handling (weighted sampling, weighted loss)
- Radiomic feature extraction
- Test-time augmentation (TTA)
- Mixup and CutMix augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import cv2
from scipy import ndimage


# ============================================
# Dataset Statistics for PneumoniaMNIST
# ============================================
# Pre-computed mean and std for PneumoniaMNIST (grayscale)
PNEUMONIA_MNIST_MEAN = [0.5]
PNEUMONIA_MNIST_STD = [0.5]


# ============================================
# Core Transforms
# ============================================
def get_train_transform(use_augmentation=True):
    """
    Get training transform with optional augmentation.
    
    Args:
        use_augmentation: Whether to apply data augmentation
        
    Returns:
        torchvision.transforms.Compose: Training transform pipeline
    """
    if use_augmentation:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(mean=PNEUMONIA_MNIST_MEAN, std=PNEUMONIA_MNIST_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=PNEUMONIA_MNIST_MEAN, std=PNEUMONIA_MNIST_STD),
        ])


def get_val_transform():
    """
    Get validation/test transform (no augmentation).
    
    Returns:
        torchvision.transforms.Compose: Validation transform pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=PNEUMONIA_MNIST_MEAN, std=PNEUMONIA_MNIST_STD),
    ])


# ============================================
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================
class CLAHETransform:
    """
    Apply CLAHE contrast enhancement.
    Works on PIL Images or numpy arrays.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        """Apply CLAHE to image."""
        # Convert PIL to numpy if needed
        if hasattr(img, 'numpy'):
            img_np = np.array(img)
        else:
            img_np = img
            
        # Convert to uint8 if needed
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
            
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=self.tile_grid_size
        )
        
        if len(img_np.shape) == 2:
            # Grayscale
            enhanced = clahe.apply(img_np)
        else:
            # Convert to LAB color space for color images
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        return enhanced


def get_train_transform_with_clahe(use_augmentation=True):
    """
    Get training transform with CLAHE preprocessing.
    
    Args:
        use_augmentation: Whether to apply data augmentation
        
    Returns:
        torchvision.transforms.Compose: Training transform with CLAHE
    """
    base_transforms = [
        CLAHETransform(clip_limit=2.0),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
    
    if use_augmentation:
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    else:
        aug_transforms = []
        
    normalize = [transforms.Normalize(mean=PNEUMONIA_MNIST_MEAN, std=PNEUMONIA_MNIST_STD)]
    
    return transforms.Compose(base_transforms + aug_transforms + normalize)


# ============================================
# Class Imbalance Handling
# ============================================
def get_class_weights(dataset):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        dataset: PyTorch dataset with labels
        
    Returns:
        torch.Tensor: Class weights inversely proportional to class frequency
    """
    labels = []
    for _, label in dataset:
        labels.append(label.item() if hasattr(label, 'item') else int(label))
    
    labels = np.array(labels)
    class_counts = np.bincount(labels.flatten().astype(int))
    total = len(labels)
    
    # Inverse frequency weighting
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def get_sample_weights(dataset):
    """
    Calculate per-sample weights for WeightedRandomSampler.
    
    Args:
        dataset: PyTorch dataset with labels
        
    Returns:
        list: Per-sample weights
    """
    labels = []
    for _, label in dataset:
        labels.append(label.item() if hasattr(label, 'item') else int(label))
    
    labels = np.array(labels).flatten().astype(int)
    class_counts = np.bincount(labels)
    
    # Weight each sample by inverse class frequency
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    return sample_weights.tolist()


def get_weighted_sampler(dataset):
    """
    Create WeightedRandomSampler for class-balanced batches.
    
    Args:
        dataset: PyTorch dataset with labels
        
    Returns:
        WeightedRandomSampler: Sampler for balanced mini-batches
    """
    sample_weights = get_sample_weights(dataset)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def create_weighted_dataloader(dataset, batch_size=64, num_workers=0):
    """
    Create DataLoader with weighted sampling for class balance.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader: DataLoader with weighted sampling
    """
    sampler = get_weighted_sampler(dataset)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers
    )


# ============================================
# Weighted Loss Functions
# ============================================
class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with class weights for imbalanced data.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            weight = self.class_weights.to(inputs.device)
            return F.cross_entropy(inputs, targets, weight=weight)
        return F.cross_entropy(inputs, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces the relative loss for well-classified examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
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


# ============================================
# Mixup and CutMix Augmentation
# ============================================
def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation to a batch.
    
    Args:
        x: Input images (batch)
        y: Labels (batch)
        alpha: Mixup interpolation strength
        
    Returns:
        tuple: Mixed inputs, label pairs, and lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute Mixup loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Label pairs from mixup
        lam: Mixup lambda value
        
    Returns:
        torch.Tensor: Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation to a batch.
    
    Args:
        x: Input images (batch)
        y: Labels (batch)
        alpha: CutMix interpolation strength
        
    Returns:
        tuple: CutMixed inputs, label pairs, and lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Get bounding box for CutMix
    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# ============================================
# Test-Time Augmentation (TTA)
# ============================================
class TestTimeAugmentation:
    """
    Apply test-time augmentation for more robust predictions.
    Averages predictions across multiple augmented versions of the input.
    """
    def __init__(self, model, device, n_augments=8):
        self.model = model
        self.device = device
        self.n_augments = n_augments
        
        # Define TTA transforms (deterministic)
        self.transforms = [
            lambda x: x,                                    # Original
            lambda x: torch.flip(x, dims=[2]),              # Horizontal flip
            lambda x: torch.flip(x, dims=[3]),              # Vertical flip
            lambda x: torch.flip(x, dims=[2, 3]),           # Both flips
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),     # 90° rotation
            lambda x: torch.rot90(x, k=2, dims=[2, 3]),     # 180° rotation
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),     # 270° rotation
            lambda x: torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[2, 3]),  # Flip + 90°
        ]
        
    def predict(self, x):
        """
        Make TTA prediction.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            torch.Tensor: Averaged softmax probabilities
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for i, transform in enumerate(self.transforms[:self.n_augments]):
                augmented = transform(x.to(self.device))
                outputs = self.model(augmented)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
                
        # Average predictions
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs


# ============================================
# Radiomic Features
# ============================================
def extract_radiomic_features(img):
    """
    Extract radiomic features from an image.
    
    Args:
        img: Numpy array image (H, W) or (H, W, C)
        
    Returns:
        dict: Dictionary of radiomic features
    """
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)  # Convert to grayscale
        
    img = img.astype(np.float64)
    
    features = {}
    
    # Intensity features
    features['mean_intensity'] = np.mean(img)
    features['std_intensity'] = np.std(img)
    features['min_intensity'] = np.min(img)
    features['max_intensity'] = np.max(img)
    features['median_intensity'] = np.median(img)
    
    # Histogram features
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    hist = hist / hist.sum()  # Normalize
    features['entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
    features['skewness'] = float(((img - np.mean(img))**3).mean() / (np.std(img)**3 + 1e-10))
    features['kurtosis'] = float(((img - np.mean(img))**4).mean() / (np.std(img)**4 + 1e-10) - 3)
    
    # Texture features (Laplacian)
    laplacian = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F)
    features['blur_score'] = laplacian.var()
    
    # Edge features (Sobel)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    features['edge_density'] = np.mean(edge_magnitude)
    features['edge_std'] = np.std(edge_magnitude)
    
    return features


def extract_batch_features(images):
    """
    Extract radiomic features from a batch of images.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        
    Returns:
        np.ndarray: Feature matrix (B, num_features)
    """
    all_features = []
    
    for i in range(images.shape[0]):
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]  # Remove channel dimension for grayscale
        else:
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            
        # Denormalize
        img = (img * 0.5 + 0.5) * 255
        img = np.clip(img, 0, 255)
        
        features = extract_radiomic_features(img)
        all_features.append(list(features.values()))
        
    return np.array(all_features)


# ============================================
# Convenience Functions
# ============================================
def setup_preprocessing_pipeline(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=64,
    use_augmentation=True,
    use_weighted_sampling=True,
    num_workers=0
):
    """
    Set up complete preprocessing pipeline with transforms and dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for dataloaders
        use_augmentation: Whether to use data augmentation
        use_weighted_sampling: Whether to use weighted sampling for class balance
        num_workers: Number of data loading workers
        
    Returns:
        dict: Dictionary containing dataloaders and preprocessing utilities
    """
    # Create dataloaders
    if use_weighted_sampling:
        train_loader = create_weighted_dataloader(
            train_dataset, 
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Calculate class weights for weighted loss
    class_weights = get_class_weights(train_dataset)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_weights': class_weights,
        'train_transform': get_train_transform(use_augmentation),
        'val_transform': get_val_transform(),
    }


def print_preprocessing_info():
    """Print information about available preprocessing options."""
    print("=" * 60)
    print("Preprocessing Pipeline for PneumoniaMNIST")
    print("=" * 60)
    print("\nAvailable Transforms:")
    print("  - get_train_transform(use_augmentation=True)")
    print("  - get_val_transform()")
    print("  - get_train_transform_with_clahe(use_augmentation=True)")
    print("\nClass Imbalance Handling:")
    print("  - get_weighted_sampler(dataset)")
    print("  - create_weighted_dataloader(dataset, batch_size)")
    print("  - WeightedCrossEntropyLoss(class_weights)")
    print("  - FocalLoss(alpha=1.0, gamma=2.0)")
    print("\nAugmentation:")
    print("  - mixup_data(x, y, alpha=0.2)")
    print("  - cutmix_data(x, y, alpha=1.0)")
    print("  - TestTimeAugmentation(model, device, n_augments=8)")
    print("\nRadiomic Features:")
    print("  - extract_radiomic_features(img)")
    print("  - extract_batch_features(images)")
    print("\nConvenience Functions:")
    print("  - setup_preprocessing_pipeline(...)")
    print("=" * 60)


if __name__ == "__main__":
    print_preprocessing_info()
