# System-Mediated Attention Imbalances Make Vision-Language Models Say Yes
*Tsan Tsai Chan*, [*Varsha Suresh*](https://sites.google.com/view/varsha-suresh/), [*Anisha Saha*](https://anisha0325.github.io/), [*Michael Hahn*](https://www.mhahn.info/), [*Vera Demberg*](https://www.uni-saarland.de/lehrstuhl/demberg/members/verademberg.html)  

## Overview

This repository accompanies our 2026 ACL Findings paper ([arXiv 2601.12430](https://arxiv.org/pdf/2601.12430)). 

The codebase implements attention intervention experiments on **LLaVA-1.5 7B**, allowing users to systematically analyse how attention weights to its three input modalities (text, image, and system prompts) contribute to model behaviour and hallucination. 

The interventions are predominantly encoded in `modeling_llama.py` (see Repository Structure).

## Abstract

Vision-language model (VLM) hallucination is commonly linked to imbalanced allocation of attention across input modalities: system, image and text. However, existing mitigation strategies tend towards an image-centric interpretation of these imbalances, often prioritising increased image attention while giving less consideration to the roles of the other modalities. In this study, we evaluate a more holistic, system-mediated account, which attributes these imbalances to functionally redundant system weights that reduce attention to image and textual inputs. We show that this framework offers a useful empirical perspective on the yes-bias, a common form of hallucination in which VLMs indiscriminately respond 'yes'. Causally redistributing attention from the system modality to image and textual inputs substantially suppresses this bias, often outperforming existing approaches. We further present evidence suggesting that system-mediated attention imbalances contribute to the yes-bias by encouraging a default reliance on coarse input representations, which are effective for some tasks but ill-suited to others. Taken together, these findings firmly establish system attention as a key factor in VLM hallucination and highlight its potential as a lever for mitigation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Dataset Preparation](#dataset-preparation)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Attention Head Data](#attention-head-data)
- [Running Interventions](#running-interventions)
  - [Text Modality Interventions](#text-modality-interventions)
  - [Image Modality Interventions](#image-modality-interventions)
  - [System Modality Interventions](#system-modality-interventions)
- [Understanding Results](#understanding-results)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Prerequisites

### Hardware Requirements
- CUDA-capable GPU with at least 16GB VRAM (60GB+ may be required for the larger datasets such as NaturalBench)
- Sufficient system RAM and free disk space for models and datasets

### Software Requirements
- Python 3.8+
- CUDA 11.7+ (for PyTorch GPU support)
- Conda or Miniconda
- Git

### HuggingFace Requirements
- HuggingFace account and API token (if using the gated dataset Winoground)

### Key Dependencies
The setup will install:
- PyTorch 2.0+
- Transformers
- LLaVA dependencies
- pycocotools
- PIL/Pillow
- tqdm, shortuuid

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ttt0524/modality_bias_2.git
cd modality_bias
```

### 2. Set Up Environment

```bash
cd LLaVA
pip install -e .
```

This will install LLaVA and all required dependencies in editable mode.

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch:  {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Configuration

### Environment Variables

The bash scripts use the following environment variables with defaults:

```bash
# Directory for data and models (default: /scratch/$USER)
export SCRATCH_DIR=/path/to/your/scratch/directory

# Conda installation path (default: $SCRATCH_DIR/miniconda3)
export CONDA_PATH=/path/to/conda

# Conda environment name (default: y25)
export CONDA_ENV=your_env_name

# Repository directory (default: $SCRATCH_DIR/Hallucination-Attribution/LLaVA)
export REPO_DIR=/path/to/modality_bias/LLaVA

# HuggingFace token (optional, for BEAF; see above)
export HUGGINGFACE_HUB_TOKEN=your_hf_token_here

# HuggingFace cache directory (optional)
export HF_HOME=$SCRATCH_DIR/. hf_cache
```

## Dataset Preparation

1. **HuggingFace datasets**: The study focusses on the six single-token-generation benchmarks BEAF, HallusionBench, MME, POPE, NaturalBench, SugarCrepe and Whoops!. The evaluation scripts, however, additionally support several other datasets, all hosted on HuggingFace. Datasets with larger numbers of prompts may be downloaded in full or in reduced form for testing.

2. **COCO dataset**: This is meant for the COCO captioning task. While not the focus of this paper, support for this dataset is retained for backwards compatibility with the original repository (see Acknowledgements).

### Model Checkpoint

The scripts automatically download the LLaVA-1.5-7B model from HuggingFace: 
- Model ID: `liuhaotian/llava-v1.5-7b`

## Repository Structure

```
modality_bias/
├── README.md
├── LLaVA/
│   ├── llava/
│   │   ├── model/
│   │   │   └── language_model/
│   │   │       └── modeling_llama.py    # Core intervention implementation
│   │   └── ... 
│   ├── eval_scripts/
│   │   ├── analyze_attention_reweight_matrices_fm_sys.py    # System intervention script
│   │   ├── analyze_attention_reweight_matrices_fm_img.py    # Image intervention script
│   │   └── analyze_attention_reweight_matrices_fm_text.py   # Text intervention script
│   ├── bash_scripts/
│   │   ├── y25_fm_sys.sh              # Shell file for text intervention script
│   │   ├── y25_fm_img.sh              # Shell file for image intervention script
│   │   └── y25_fm_text.sh             # Shell file for system intervention script
│   └── results/
│       └── coco/
│           └── llava_3000/
│               └── identify_attention_head/
│                   ├── all_heads_llava157b.json        # All identified heads
│                   ├── layers1to8_llava157b.json       # Layer 1-8 heads
│                   ├── layers9to16_llava157b.json      # Layer 9-16 heads
│                   ├── layers17to24_llava157b.json     # Layer 17-24 heads
│                   ├── layers25to32_llava157b.json     # Layer 25-32 heads
│                   └── attribution_result.json         # Attribution scores
```

## Pipeline Overview

The intervention pipeline consists of three main components: 

### 1. Attention Intervention Mechanism (`modeling_llama.py`)

This file contains the modified LLaMA attention implementation that enables several different runtime interventions on attention weights.  It allows: 
- Pairwise and proportional redistribution of attention weights across modalities
- Selective up- and downweighting of attention to specific modalities (system, image, text) for backwards compatibility 
- Control over which of LLaVA-1.5 7B's decoder heads to apply interventions to

### 2. Selecting Attention Heads for Intervention (JSON Files)

Pre-identified attention heads: 

- **`all_heads_llava157b.json`**: Complete list of all attention heads across all 32 layers of LLaVA-1.5-7B
- **Layer-specific files** (`layers1to8_llava157b.json`, etc. ): Heads in specific layers
- **`attribution_result.json`**: The 30 heads identified by Yang et al. (ICLR 2025) as contributing most to object hallucination in the COCO image captioning task

These JSON files specify which attention heads to intervene on during evaluation. Each file contains zero-indexed layer and head indices, with only the heads listed under 'hal_heads' being intervened on:

```json
{
  "hal_heads": [
    [layer_idx, head_idx],
    [10, 5],
    [15, 3]
  ],
  "non_hal_heads": [
    [layer_idx, head_idx],
    ... 
  ]
}
```

### 3. Evaluation Scripts

Three Python scripts that implement the evaluation pipeline for each modality:

- **`analyze_attention_reweight_matrices_fm_sys.py`**: Downscales system attention
- **`analyze_attention_reweight_matrices_fm_img.py`**: Downscales image attention
- **`analyze_attention_reweight_matrices_fm_text.py`**: Downscales text attention

Each script:
1. Loads the LLaVA model
2. Processes the selected dataset
3. Applies attention downweighting to specified heads
4. Evaluates model performance
5. Saves model outputs and per-head attention matrices for analysis

Upscaling modality weights is supported by the modified `modeling_llama.py` in this repository for backwards compatibility, but not coupled to an evaluation script.

## Running Interventions

All intervention scripts are located in `LLaVA/bash_scripts/` and should be run from the repository root.

### General Command Structure

```bash
bash bash_scripts/y25_fm_<modality>. sh <dataset> <alpha> <redistribute_to> <use_reduced>
```

### Parameters

| Parameter | Description | Valid Values | Example |
|-----------|-------------|--------------|---------|
| `dataset` | Dataset to evaluate on | `beaf`, `hallusionbench`, etc. | `beaf` |
| `alpha` | Reweighting factor | Float from 0.0 to 1.0 | `0.5`, `1.0` |
| `redistribute_to` | Which modality to redistribute attention to | `system`, `img`, `text` | `img` |
| `use_reduced` | Use reduced head set | `True`, `False` | `True` |

**Alpha values**:
- `< 1.0`: Downscale attention (reduce influence of modality weights are removed from)
- `= 1.0`: No change (baseline)
- `= 0.0`: Zero-ablation of attention (remove influence)

**Redistribute_to**:
- Specifies which modality/ modalities receive(s) the redistributed attention weight
- When downscaling text with `redistribute_to = img`, removed text attention is added to image attention

**Use_reduced**:
- `True`: Only load subsets of selected HuggingFace datasets to reduce computational load; subsets are stored separately from the full versions on HuggingFace
- `False`: Load full dataset


## Outputs

- JSON file containing model predictions
- CSV file with per-head attention patterns per prompt

Every item in each file lists the model response alongside the ground-truth answer.

## Acknowledgements

This code is modified from the paper **"Understanding and Mitigating Hallucinations in Large Vision-Language Models via Modular Attribution and Intervention"** (ICLR 2025).

Original codebase: [https://github.com/TianyunYoung/Hallucination-Attribution](https://github.com/TianyunYoung/Hallucination-Attribution) 

The LLaVA model and framework:  [[https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)](https://github.com/haotian-liu/LLaVA)

The work of these codebases' authors is gratefully acknowledged.

---

## Citation

If you use this code, please cite the original paper: 

```bibtex
@article{chan2026system,
  title={System-Mediated Attention Imbalances Make Vision-Language Models Say Yes},
  author={Chan, Tsan Tsai and Suresh, Varsha and Saha, Anisha and Hahn, Michael and Demberg, Vera},
  journal={arXiv preprint arXiv:2601.12430},
  year={2026}
}
```
