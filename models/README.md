# Model Files

This directory should contain model files for the car damage assessment system.

## Required Models

- `fast_model.h5` - MobileNetV2-based model for car damage classification

## How to Obtain Models

Due to GitHub file size limitations, model files are not included in this repository. You can:

1. **Train your own models** using the training scripts provided in this repository
2. **Download pre-trained models** from a separate storage location (if available)
3. **Use Git LFS** if you're setting up your own repository with these files

## Model Details

The default model architecture is MobileNetV2 with the following specifications:
- Input size: 160x160 pixels
- Output classes: 4 (minor damaged car, moderate car damaged, normal, severe car damaged)
- Training accuracy: ~82% on test set
- Base model: MobileNetV2 with transfer learning 