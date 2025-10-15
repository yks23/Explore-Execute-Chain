#!/bin/bash
#
# E2C-SFT Training Script
# This script trains the E2C model using supervised fine-tuning (SFT)
# on exploration-execution paired data.
# Can be run from scripts directory
#
# Usage:
#   1. Run with auto-detected GPUs (recommended):
#      bash e2c_sft.sh
#
#   2. Or specify GPUs manually:
#      export CUDA_VISIBLE_DEVICES="0,1"
#      bash e2c_sft.sh
#
#   3. Customize other parameters:
#      export MODEL_PATH="Qwen/Qwen3-8B"
#      export TOTAL_TRAINING_STEPS=1000
#      bash e2c_sft.sh
#
# Note: GPU devices are auto-detected if CUDA_VISIBLE_DEVICES is not set
#

# ============================================================================
# Get script directory and project root
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root for running training
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration - Can be overridden by environment variables
# ============================================================================

# Model path - Use HuggingFace model ID (will auto-download)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

# Data paths (from project root)
TRAIN_DATA="${TRAIN_DATA:-data/processed/sft/e2c-sft-train.parquet}"
VAL_DATA="${VAL_DATA:-data/processed/sft/e2c-sft-val.parquet}"

# Output directory (from project root)
OUTPUT_DIR="${OUTPUT_DIR:-models/checkpoints/sft}"

# Training configuration
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"  # 1000 iterations is sufficient
PROJECT_NAME="${PROJECT_NAME:-e2c-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-run1}"

# GPU configuration - Auto-detect available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Auto-detect all available GPUs
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        if [ "$GPU_COUNT" -gt 0 ]; then
            CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
            CUDA_NUM=$GPU_COUNT
            echo "Auto-detected $GPU_COUNT GPUs: $CUDA_VISIBLE_DEVICES"
        else
            echo "Warning: No GPUs detected, using default: 0,1,2,3"
            CUDA_VISIBLE_DEVICES="0,1,2,3"
            CUDA_NUM=4
        fi
    else
        echo "Warning: nvidia-smi not found, using default: 0,1,2,3"
        CUDA_VISIBLE_DEVICES="0,1,2,3"
        CUDA_NUM=4
    fi
else
    # Count GPUs from CUDA_VISIBLE_DEVICES
    CUDA_NUM=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "Using specified GPUs: $CUDA_VISIBLE_DEVICES (count: $CUDA_NUM)"
fi

MASTER_PORT="${MASTER_PORT:-12345}"

# ============================================================================
# Validate Configuration
# ============================================================================

echo "=========================================="
echo "E2C-SFT Training Configuration"
echo "=========================================="
echo "Model:             $MODEL_PATH"
echo "Training Data:     $TRAIN_DATA"
echo "Validation Data:   $VAL_DATA"
echo "Output Directory:  $OUTPUT_DIR"
echo "Training Steps:    $TOTAL_TRAINING_STEPS"
echo "GPU Devices:       $CUDA_VISIBLE_DEVICES"
echo "GPU Count:         $CUDA_NUM"
echo "Master Port:       $MASTER_PORT"
echo "=========================================="
echo ""
echo "Note: Model will be automatically downloaded from HuggingFace if not cached."
echo ""

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo "Please prepare the data first:"
    echo "  bash scripts/prepare_all_data.sh"
    exit 1
fi

# Check if validation data exists
if [ ! -f "$VAL_DATA" ]; then
    echo "❌ Error: Validation data not found at $VAL_DATA"
    echo "Please prepare the data first:"
    echo "  bash scripts/prepare_all_data.sh"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Start Training
# ============================================================================

echo "Starting E2C-SFT training..."
echo ""

export CUDA_VISIBLE_DEVICES
export CUDA_NUM
export MASTER_PORT

# Add verl to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH}"

torchrun --nproc_per_node $CUDA_NUM "./verl/verl/trainer/fsdp_sft_trainer.py" \
  --config-path config \
  --config-name sft_trainer_2 \
  model.partial_pretrain="$MODEL_PATH" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  trainer.n_gpus_per_node=$CUDA_NUM

# ============================================================================
# Training Complete
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ E2C-SFT Training Complete!"
    echo "=========================================="
    echo "Checkpoint saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Verify checkpoint:"
    echo "     ls -lh $OUTPUT_DIR"
    echo ""
    echo "  2. Start RL training:"
    echo "     export MODEL_PATH=\"$OUTPUT_DIR/final\""
    echo "     bash scripts/e2c_rl.sh"
    echo "=========================================="
else
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    exit 1
fi

