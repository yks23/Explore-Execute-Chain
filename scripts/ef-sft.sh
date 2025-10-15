#!/bin/bash
#
# EF-SFT (Exploration-Focused SFT) Training Script
# This script adapts E2C models to new domains using exploration-only data.
# It requires only ~3.5% of training tokens compared to full SFT.
# Can be run from scripts directory
#
# Usage:
#   1. Run from scripts directory:
#      bash ef-sft.sh
#
#   2. Or customize with environment variables:
#      export MODEL_PATH="models/checkpoints/rl/stage2-main/final"
#      bash ef-sft.sh
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

# Model path - Should be E2C trained model or HF model ID
# Default: use local RL checkpoint, or set to HF model
MODEL_PATH="${MODEL_PATH:-models/checkpoints/rl/stage2-main/final}"

# Data paths (from project root)
TRAIN_DATA="${TRAIN_DATA:-data/processed/ef_sft/domain-train.parquet}"
VAL_DATA="${VAL_DATA:-data/processed/ef_sft/domain-val.parquet}"

# Output directory (from project root)
OUTPUT_DIR="${OUTPUT_DIR:-models/checkpoints/ef_sft}"

# Training configuration
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"  # 500 iterations is sufficient for EF-SFT
PROJECT_NAME="${PROJECT_NAME:-ef-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-run1}"

# GPU configuration
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CUDA_NUM="${CUDA_NUM:-4}"
MASTER_PORT="${MASTER_PORT:-12345}"

# ============================================================================
# Validate Configuration
# ============================================================================

echo "=========================================="
echo "EF-SFT (Domain Adaptation) Configuration"
echo "=========================================="
echo "Model Path:        $MODEL_PATH"
echo "Training Data:     $TRAIN_DATA"
echo "Validation Data:   $VAL_DATA"
echo "Output Directory:  $OUTPUT_DIR"
echo "Training Steps:    $TOTAL_TRAINING_STEPS"
echo "GPUs:              $CUDA_NUM"
echo "=========================================="
echo ""
echo "📌 Note: EF-SFT adapts the exploration strategy to new domains"
echo "   with minimal data (only ~3.5% of full SFT tokens)."
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "Please complete RL training first, or use a pre-trained E2C model:"
    echo "  bash scripts/e2c_rl.sh"
    echo ""
    echo "Or download a pre-trained E2C model from HuggingFace"
    exit 1
fi

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo ""
    echo "EF-SFT requires exploration-only data for your target domain."
    echo "Please prepare domain-specific exploration data."
    echo ""
    echo "Data format: Each sample should contain:"
    echo "  - prompt: The problem/question"
    echo "  - exploration: Reasoning plan (lightweight sketch)"
    echo ""
    echo "See main README.md for more details on preparing EF-SFT data."
    exit 1
fi

# Create validation data check (optional for EF-SFT)
if [ ! -f "$VAL_DATA" ]; then
    echo "⚠️  Warning: Validation data not found at $VAL_DATA"
    echo "   Proceeding without validation (not recommended)"
    VAL_DATA=""
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Start Training
# ============================================================================

echo "Starting EF-SFT domain adaptation..."
echo ""

export CUDA_VISIBLE_DEVICES
export CUDA_NUM
export MASTER_PORT

# Add verl to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH}"

# Build validation file argument if exists
VAL_ARG=""
if [ -n "$VAL_DATA" ]; then
    VAL_ARG="data.val_files=\"$VAL_DATA\""
fi

torchrun --nproc_per_node $CUDA_NUM "./verl/verl/trainer/fsdp_sft_trainer.py" \
  --config-path config \
  --config-name sft_trainer_2 \
  model.partial_pretrain="$MODEL_PATH" \
  data.train_files="$TRAIN_DATA" \
  $VAL_ARG \
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
    echo "✓ EF-SFT Domain Adaptation Complete!"
    echo "=========================================="
    echo "Checkpoint saved to: $OUTPUT_DIR"
    echo ""
    echo "Your E2C model has been adapted to the new domain!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify checkpoint:"
    echo "     ls -lh $OUTPUT_DIR"
    echo ""
    echo "  2. Test on domain-specific problems:"
    echo "     export MODEL_PATH=\"$OUTPUT_DIR/final\""
    echo "     bash scripts/eval.sh --dataset your_domain_test"
    echo ""
    echo "  3. Run interactive demo:"
    echo "     python example_interactive.py --model-path \"$OUTPUT_DIR/final\""
    echo "=========================================="
else
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    exit 1
fi

