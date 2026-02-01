# Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/Conference-ICLR_2026-blue)](https://iclr.cc/)

> **🚀 Resolving the Fidelity-Speed Dilemma in Implicit Neural Representations (INRs)**

This repository contains the official implementation for the paper **"Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields"**, accepted to **ICLR 2026**.

### 📄 Abstract
Implicit Neural Representations (INRs) often force a compromise: high-fidelity MLP models are slow to query, while fast embedding-based models struggle with expressivity. We propose Decoupled Representation Refinement (DRR), an architectural paradigm that resolves this by learning deep, expressive feature transformations directly on the embedding structure. By utilizing a one-time, offline process to encode rich, non-linear representations into the embeddings, we effectively decouple the heavy network from the inference path. This enables the fast embedding query to deliver highly expressive features, achieving high representational capacity with minimal inference latency.

### ✨ Key Features
* **⚡ State-of-the-Art Accuracy & Efficiency** Achieves state-of-the-art generalization for unseen input conditions and coordinates for large-scale 3D volumetric simulations. It delivers high-fidelity reconstructions while being being **27$\times$ faster** with **45$\times$ reduction** in computation (TFLOPs per billion queries) compared to high-fidelity baselines.
* **🧠 Decoupled Representation Refinement (DRR):** Separates the expensive learning phase from the fast query phase, matching the speed of pure embedding-based methods.
* **📈 Variational Pairs (VP):** A novel data augmentation strategy designed for INR tasks, demonstrated improvement on generalization for high-dimensional surrogate modeling and sparse ensemble data.

---

## 1. Project Structure

The repository is organized as follows:

```
├── configs/              # Configuration files for all experiments
├── datasets/             # Dataset implementations
├── models/               # Model implementations
├── util/                 # General utility functions
├── train.py              # Main script for training models
├── test.py               # Main script for evaluating models
├── test_flops.py         # Script for calculating TFLOPs / 10^9 points
├── train_ngp_nerf_occ.py # Training and evaluating embedding-based NeRF models
├── train_mlp_nerf.py     # Training and evaluating the original MLP-based NeRF
├── start_jobs.py         # Optional helper script to run experiments
└── requirements.txt
```

## 2. Environment Setup

The experiments were done with `Python 3.11` and `PyTorch 2.5.1` with `CUDA 12.4`. We recommend using a virtual environment to manage dependencies. You can set up the environment using either `pip` or `conda`. Example for `conda`:

```bash
# Create and activate a conda environment
conda create --name drr-inr python=3.11
conda activate drr-inr

# Install the required packages
pip install -r requirements.txt
```

For running NeRF-related experiments, additionally install tiny-cuda-nn with:
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

---
## 3. Running Experiments

This project uses [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) for training and evaluation. While current setup works for distributed training, all experiments in the paper were conducted on a single GPU using the provided configuration file at `configs/accl/one.yaml`.

### 3.1. General Execution Commands

The primary scripts for most experiments are `train.py` and `test.py`.

* **Training:**
    ```bash
    accelerate launch --config_file configs/accl/one.yaml train.py \
        --config [PATH_TO_EXPERIMENT_CONFIG] \
        --config_id [CONFIG_ID]
    ```

* **Evaluation:**
    ```bash
    accelerate launch --config_file configs/accl/one.yaml test.py \
        --config [PATH_TO_EXPERIMENT_CONFIG] \
        --config_id [CONFIG_ID]
    ```

### 3.2. Experiment Configuration Guide

The configuration files for all experiments are located in the `configs/` directory.

#### **Ensemble Simulation Surrogate Evaluations**

* **Condition Generalization:**
    * **Config Path:** `configs/{dataset}/{model}.yaml`
    * **`{dataset}`:** `nyx`, `mpas_graph`, or `clover_600`.
    * **`{model}`:** `drrnet` (our model) or `baselines`.
    * In `baselines.yaml`, the configurations correspond to **K-Plane** (`config_id=0`), **Explorable-INR** (`config_id=1`), and **FA-INR** (`config_id=2`).

* **Spatio-Condition Generalization:**
    * **Config Path:** `configs/{dataset}/spatio_cond_gen/{model}.yaml`

#### **Neural Image Representation**

* **Config Path:** `configs/image/{dataset}.yaml`
* **`{dataset}`:** `pearl` or `tokyo`.
* Each file contains the configurations for all models evaluated on that image.

#### **Neural Radiance Fields (NeRF)**

The NeRF experiments use dedicated training scripts.

* **For embedding-based and our DRR-enhanced models:**
    ```bash
    python train_ngp_nerf_occ.py \
        --config [PATH_TO_NeRF_CONFIG] \
        --config_id [CONFIG_ID]
    ```
* **For the original MLP-based NeRF model:**
    ```bash
    python train_mlp_nerf.py \
        --config [PATH_TO_NeRF_CONFIG] \
        --config_id 3
    ```

---

## 4. Dataset Preparation (`HDFieldDataset`)

See the dataset preparation instructions in [datasets/README.md](datasets/README.md).