#!/bin/bash
set -euo pipefail

# -------------------------------
# Set these paths before running
# -------------------------------
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$USER}
CONDA_PATH=${CONDA_PATH:-$SCRATCH_DIR/miniconda3}
CONDA_ENV=${CONDA_ENV:-y25}
REPO_DIR=${REPO_DIR:-$SCRATCH_DIR/Hallucination-Attribution/LLaVA}

# -------------------------------
# Positional arguments from HTCondor
# -------------------------------
dataset=$1
system_alpha=$2
redistribute_to=$3
use_reduced=$4

# -------------------------------
# Conda run environment
# -------------------------------
PY=$CONDA_PATH/envs/$CONDA_ENV/bin/python

# Minimal Python environment
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export PYTHONSTARTUP=""
export PYTHONDONTWRITEBYTECODE=1
export LANG=C
export LC_ALL=C
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
exec 0</dev/null   # disconnect stdin

# -------------------------------
# Hugging Face setup
# -------------------------------
unset HF_HUB_OFFLINE
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-"YOUR_TOKEN_HERE"}
export HF_HOME=${HF_HOME:-$SCRATCH_DIR/.hf_cache}

# -------------------------------
# Switch to working directory
# -------------------------------
cd $SCRATCH_DIR/Hallucination-Attribution/LLaVA
echo "Working directory: $(pwd)"
echo "Dataset: $dataset"
echo "System Alpha: $system_alpha"
echo "Redistribute To: $redistribute_to"
echo "Use Reduced: $use_reduced"

# -------------------------------
# Run LLaVA evaluation
# -------------------------------
echo "===== RUNNING EVALUATION ====="
cmd=(
    conda run -n $CONDA_ENV --no-capture-output python -m eval_scripts.analyze_attention_reweight_matrices_fm_sys
    --dataset "$dataset"
    --model-path liuhaotian/llava-v1.5-7b
    --attention_head_path ./results/coco/llava_3000/identify_attention_head/all_heads_llava157b.json
    --reweight_system
    --system_alpha "$system_alpha"
    --redistribute_to "$redistribute_to"
)

if [ "$use_reduced" = "True" ]; then
    cmd+=(--use_reduced)
fi

# Run the evaluation
"${cmd[@]}"
