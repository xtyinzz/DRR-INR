# Official Code for *Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields*

This repository contains the official implementation for the paper **"Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields"** (Submission ID: **8494**).

This document provides instructions on how to set up the environment and execute the experiments required to reproduce the main results reported in our paper.

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

This project uses [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) for training and evaluation. While `accelerate` simplifies scaling to multi-GPU setups, all experiments in the paper were conducted on a single GPU using the provided configuration file at `configs/accl/one.yaml`.

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