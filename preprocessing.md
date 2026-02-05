## Preprocessing Ideas for PathMNIST Binary Classification

### 1. Core Tensor Transforms

- **Training transform (with augmentation)**:

```python
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Normalize(mean=[0.7406, 0.5331, 0.7059],
                         std=[0.1194, 0.1570, 0.1141]),
])
```

- **Validation / test transform (no augmentation)**:

```python
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7406, 0.5331, 0.7059],
                         std=[0.1194, 0.1570, 0.1141]),
])
```

### 2. Radiomic Feature Ideas

- **Existing**:
  - Mean intensity
  - Standard deviation of intensity
  - Blur score (Laplacian variance)

- **Additional features to explore**:
  - **Texture (GLCM)**: contrast, correlation, energy, homogeneity
  - **Histogram**: skewness, kurtosis, entropy, percentiles (e.g., 10th/50th/90th)
  - **Edge/shape**: edge density (Sobel/Canny), simple shape descriptors
  - **Color**: per-channel means/stds, color histogram features
  - **Frequency**: FFT or wavelet-based energy in different bands

### 3. Contrast / Stain Handling

- **CLAHE (local contrast enhancement)** on L-channel in LAB space.
- **Simple stain / color normalization**:
  - Normalize each image to target per-channel mean and std.
  - Optionally match statistics to a reference slide.

### 4. Regularization Augmentations

- **Mixup**:
  - Create convex combinations of image pairs and labels to smooth the decision boundary.

- **CutMix**:
  - Replace random patches between images and mix labels accordingly.

### 5. Test-Time Augmentation (TTA)

- At inference:
  - Apply several deterministic transforms (e.g., flips and 90° rotations).
  - Run the model on each view.
  - Average predicted probabilities across views.

### 6. Class Imbalance Handling

- **Weighted loss**:
  - Use `BCEWithLogitsLoss(pos_weight=...)` with higher weight on the minority class (cancer).
- **Weighted sampling**:
  - `WeightedRandomSampler` so that batches are more class-balanced.

### 7. Priority Order to Try

1. Add **normalization** and **basic augmentations** (flips, rotations) to the main CNN pipeline.
2. Expand **radiomic features** and retrain the Random Forest.
3. Add **CLAHE / stain normalization** if staining variation seems high.
4. Experiment with **Mixup / CutMix** and **class weighting** if metrics (e.g., recall for cancer) are poor.
5. Use **TTA** when reporting final test metrics.

