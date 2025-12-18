# Animal-6 Image Classification using CNN

This project implements a simplified convolutional neural network for classifying 6 animal categories.

## Dataset
Animal-6 dataset containing:
- butterfly
- cat
- chicken
- elephant
- horse
- spider

## Model Architecture
- Conv2D (16 filters)
- MaxPool
- Conv2D (32 filters)
- MaxPool
- Dense → Softmax

## How to Run

```bash
pip install tensorflow numpy matplotlib
python animal6_cnn.py
Results

Achieved ~70% accuracy on validation set.
See training curves or confusion matrix for more details.

Files

animal6_cnn.py — Training + testing script
