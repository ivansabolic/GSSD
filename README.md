<div align="center">

# **[Backdoor Defense through Self-Supervised and Generative Learning (BMVC 2024)](https://arxiv.org/abs/2409.01185)**
<!-- ### **[(BMVC 2024)](https://arxiv.org/abs/2409.01185)** -->

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2409.01185-b31b1b.svg)](https://arxiv.org/abs/2409.01185)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## Overview

We propose GSSD, a backdoor defense framework that leverages self-supervised and generative learning to detect and remove poisoned samples from training data. The pipeline first trains a self-supervised model on the (potentially poisoned) training set, then fits per-class normalizing flows on the extracted features to estimate sample likelihoods. Samples with low likelihood under their assigned class are flagged as suspicious, enabling effective separation of poisoned from clean data. A defended classifier is then retrained on the cleansed dataset and fine-tuned with relabeled suspicious samples.

## Installation

Requires Python >=3.10 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Pipeline

The full pipeline can be run end-to-end with the provided experiment script:

```bash
bash run_scripts/run_cifar10_experiments.sh [GPU_ID]
```

This reproduces the CIFAR-10 results from the paper (Table 1) for BadNets, Blended, and WaNet attacks.
Please reach out for attack data in order to reproduce other attacks. 

## Project Structure

```
├── train_simclr.py                  # Step 1: SimCLR training
├── extract_features_from_simclr.py  # Step 2: Feature extraction
├── train_features.py                # Step 3: Normalizing flow training
├── extract_indices_from_nflows.py   # Step 4: Poisoned sample detection
├── retraining.py                    # Steps 5-6: Retraining & fine-tuning
├── config/                          # YAML configuration files
├── run_scripts/                     # Shell scripts for full pipeline
├── simclr/                          # SimCLR model & augmentations
├── nflow/                           # Normalizing flow architectures
├── datasets/                        # Dataset loading & attack implementations
│   └── attacks/                     # Attack implementations
├── models/                          # ResNet classifiers
└── utils/                           # Utility functions
```

## Acknowledgements

Attack implementations are based on [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox).

## Citation

```bibtex
@inproceedings{Sabolic_2024_BMVC,
author    = {Ivan Sabolic and Ivan Grubišić and Siniša Šegvić},
title     = {Backdoor Defense through Self-Supervised and Generative Learning},
booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
publisher = {BMVA},
year      = {2024},
url       = {https://papers.bmvc2024.org/0346.pdf}
}
```
