#!/bin/bash
set -euo pipefail

# -------------------------------
# Set these paths before running
# -------------------------------
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$USER}
CONDA_PATH=${CONDA_PATH:-$SCRATCH_DIR/miniconda3}
CONDA_ENV=${CONDA_ENV:-y25}
REPO_DIR=${REPO_DIR:-$SCRATCH_DIR/Hallucination-Attribution/LLaVA}

# Hugging Face token (optional, only required for gated datasets like BEAF)
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-""}

# Load user config if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/config.local.env" ]] && source "$SCRIPT_DIR/config.local.env"

# -------------------------------
# Positional arguments from HTCondor
# -------------------------------
dataset=$1
text_alpha=$2
redistribute_to=$3
use_reduced=$4

# -------------------------------
# Activate conda environment
# -------------------------------
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO_DIR"

# Debugging: Print arguments
echo "Working directory: $(pwd)"
echo "Dataset: $dataset"
echo "Text Alpha: $text_alpha"
echo "Redistribute To: $redistribute_to"
echo "Use Reduced: $use_reduced"

# -------------------------------
# Run evaluation
# -------------------------------
cmd=(
  python -m eval_scripts.analyze_attention_reweight_matrices_fm_text
  --dataset "$dataset"
  --model-path liuhaotian/llava-v1.5-7b
  --attention_head_path ./results/coco/llava_3000/identify_attention_head/all_heads_llava157b.json
  --reweight_text
  --text_alpha "$text_alpha"
  --redistribute_to "$redistribute_to"
)

if [ "$use_reduced" = "True" ]; then
  cmd+=(--use_reduced)
fi

"${cmd[@]}"