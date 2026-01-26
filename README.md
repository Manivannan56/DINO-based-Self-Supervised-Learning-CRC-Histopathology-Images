# Self-Supervised Learning for Colorectal Cancer Pathology

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of DINO self-supervised learning for colorectal cancer histopathology image analysis using Vision Transformers.

## Overview

This project applies DINO (self-DIstillation with NO labels) to learn visual representations from the CRC-100K colorectal cancer dataset. We leverage Lunit's pretrained pathology weights for domain adaptation to CRC-specific features.

**Key Contributions:**
- 🔬 Implemented DINO framework for histopathology
- 🧬 Domain adaptation from general pathology (19M patches) to CRC (100K images)
- 📊 Feature analysis showing clear tissue-type clustering
- ⚡ HPC-optimized training pipeline

## Results

<p align="center">
  <img src="assets/pca_features.png" alt="PCA Visualization" width="700"/>
</p>

**PCA visualization** of learned features demonstrates clear clustering across 9 colorectal tissue types (tumor, stroma, normal mucosa, muscle, lymphocytes, adipose, mucus, debris, background).

## Quick Start

```bash
# Install dependencies
pip install torch torchvision timm numpy pillow scikit-learn matplotlib

# Download Lunit pretrained weights
wget https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch16_ep200.torch -O lunit_vit_small_dino.pth

# Download CRC-100K dataset
wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
unzip NCT-CRC-HE-100K.zip

# Extract features
python extract_features.py --data_dir ./NCT-CRC-HE-100K
```

## Dataset

**CRC-100K**: 100,000 colorectal cancer H&E images (224×224, 20× magnification)

<p align="center">
  <img src="assets/sample_images_per_class.png" alt="Sample Images per Class" width="700"/>
</p>

**9 Tissue Classes:**
- **ADI** (Adipose): Fat tissue - 10,407 images
- **BACK** (Background): Empty/artifacts - 10,566 images
- **DEB** (Debris): Tissue debris - 11,512 images
- **LYM** (Lymphocytes): Immune cells - 11,557 images
- **MUC** (Mucus): Mucus secretions - 8,896 images
- **MUS** (Muscle): Smooth muscle - 13,536 images
- **NORM** (Normal): Normal colon mucosa - 8,763 images
- **STR** (Stroma): Cancer-associated stroma - 10,446 images
- **TUM** (Tumor): Adenocarcinoma epithelium - 14,317 images

**Source**: [Zenodo](https://zenodo.org/record/1214456)

## Model

**Architecture**: Vision Transformer Small (ViT-S/16)
- Backbone: 384-dim embeddings (Lunit pretrained on 19M pathology patches)
- Projection: 384 → 2048 → 256 → 16,384 dimensions
- Training: Self-supervised knowledge distillation

**Pathology-Specific Adaptations:**
- Normalization: `mean=[0.703, 0.536, 0.661], std=[0.217, 0.261, 0.207]`
- Augmentation: H&E-aware color jitter, no grayscale
- Multi-crop: 2 global (224×224) + 8 local (96×96) views

## Usage

### Feature Extraction
```python
from model import build_dino_model

# Load pretrained model
student, teacher = build_dino_model(pretrained=True)
features = student.backbone(images)  # Extract 384-dim features
```

### Training (Continued SSL)
```python
python train.py \
  --data_dir ./NCT-CRC-HE-100K \
  --batch_size 256 \
  --lr 0.0003 \
  --epochs 30
```

### Visualization
```python
python visualize.py --features features.pt --method pca
```

## Key Findings

**Challenges in Small-Scale SSL:**
- Dataset scale matters: 100K vs 1-20M in papers
- Multi-crop strategy critical for pathology (captures cell-level features)
- Pretrained initialization essential for stability on limited data
- Hyperparameter sensitivity increases with smaller datasets

**Training Insights:**
- Random initialization and ImageNet Initialization: Unstable, diverges after epoch 5
- Global-crops-only: Challenging without multi-crop on small datasets

## Repository Structure

```
├── model.py                 # DINO architecture
├── train.py                 # SSL training script
├── dino_dataloader.py      # Data loading & augmentation
├── extract_features.py      # Feature extraction
├── visualize.py            # PCA/UMAP visualization
├── dino_train.sh           # SLURM batch script
└── README.md
```

## Citation

```bibtex
@inproceedings{kang2023benchmarking,
  title={Benchmarking Self-Supervised Learning on Diverse Pathology Datasets},
  author={Kang, Mingu and Song, Heon and Park, Seonwook and Yoo, Donggeun and Pereira, Sérgio},
  booktitle={CVPR},
  year={2023}
}

@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and Jégou, Hervé and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={ICCV},
  year={2021}
}
```

## Acknowledgments

- **Lunit Inc.**: Pretrained DINO weights ([GitHub](https://github.com/lunit-io/benchmark-ssl-pathology))
- **DINO**: Original framework ([Paper](https://arxiv.org/abs/2104.14294))
- **CRC-100K**: Dataset ([Zenodo](https://zenodo.org/record/1214456))

## License

MIT License - See LICENSE file for details

---

**Built with PyTorch • Trained on NVIDIA H200 • HPC Optimized**
