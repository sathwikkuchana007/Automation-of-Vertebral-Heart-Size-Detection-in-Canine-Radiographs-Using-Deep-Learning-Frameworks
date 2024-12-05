# Automation of Vertebral Heart Size Detection in Canine Radiographs Using Deep Learning Frameworks

This repository contains the implementation and research code for the paper **"Automation of Vertebral Heart Size Detection in Canine Radiographs Using Deep Learning Frameworks"** by Sathwik Kuchana. The study introduces a hybrid deep learning framework that combines Vision Transformers (ViT) and Convolutional Neural Networks (CNNs) to automate Vertebral Heart Size (VHS) measurements, enhancing accuracy, scalability, and reproducibility in veterinary diagnostics.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [Contact](#contact)

---

## Introduction

Vertebral Heart Size (VHS) is a key diagnostic metric for assessing canine cardiomegaly. Traditional manual methods for measuring VHS are time-consuming, labor-intensive, and prone to variability. This project leverages a hybrid deep learning framework combining ViT’s ability to capture global contextual relationships and CNN’s strength in fine-grained local feature extraction, introducing an orthogonal layer for enhanced geometric consistency.

---

## Dataset

The dataset comprises **2,000 annotated canine thoracic radiographs**, categorized into small, normal, and large heart sizes. Annotations were provided by veterinary experts, marking key anatomical landmarks for VHS measurement.

### Preprocessing Steps
- **Normalization**: Standardized pixel intensities across all images.
- **Augmentation**: Applied random rotations, flipping, and brightness/contrast adjustments.
- **Few-Shot Learning**: Used pre-trained ResNet for initial annotations, refined by experts.

---

## Model Architecture

The proposed model, RVT (Regressive Vision Transformer), combines:
1. **Vision Transformer Backbone (ViT)**: Captures global spatial relationships.
2. **Orthogonal Layer**: Ensures geometric consistency for VHS calculations.
3. **Fully Connected Layers**: Refines features for accurate key-point regression.

---

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch
- NumPy
- Matplotlib
- OpenCV

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training
To train the model, run:
```bash
python train.py
```

### Evaluation
To evaluate the model on test data, run:
```bash
python evaluate.py --model_path saved_models/vht_model.pth
```

---

## Results

### Performance Metrics
- **Mean Intersection over Union (IoU)**: 0.88
- **Mean Average Precision (mAP)**: 0.90
- **Test Accuracy**: 79%

### Highlights
- Robust generalization across diverse radiographic conditions.
- Geometric consistency ensured by the orthogonal layer.
- Significant improvements over baseline methods.

---

## Future Work

- **Semi-Supervised Learning**: Leverage unlabeled data to reduce dependency on manual annotations.
- **Multi-Species Datasets**: Expand applicability to other species.
- **Real-Time Deployment**: Optimize for real-time veterinary diagnostics.

---

## References

1. [Dosovitskiy et al., 2020: An Image is Worth 16x16 Words: Transformers for Image Recognition](https://arxiv.org/abs/2010.11929)
2. [Huang et al., 2017: Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
3. [Tan and Le, 2019: EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)

For detailed insights, refer to the [research paper](https://www.researchgate.net/publication/386425457).

---

## Contact

For questions or collaboration:
- **Author**: Sathwik Kuchana
- **Email**: skuchana@mail.yu.edu
- **Institution**: Yeshiva University
