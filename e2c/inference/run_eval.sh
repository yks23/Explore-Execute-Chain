#!/bin/bash
#
# E2C Model Complete Evaluation Pipeline (Generate + Evaluate)
# This script runs the full evaluation in two stages:
#   1. Generate model outputs
#   2. Evaluate generated outputs
#
# Usage:
#   1. Basic evaluation (GSM8K):
#      bash run_eval.sh
#
#   2. Evaluate on specific dataset:
#      bash run_eval.sh --dataset math
#
#   3. Evaluate with custom model:
#      bash run_eval.sh --model models/checkpoints/rl/stage2-main/final
#
#   4. Evaluate on all math benchmarks:
#      bash run_eval.sh --dataset all --sample 4
#
#   5. Use HuggingFace model:
#      bash run_eval.sh --model <your-org>/Explore-Execute-Chain --subfolder 4B-Final

set -e  # Exit on any error

# ============================================================================
# Get script directory and project root
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# ============================================================================
# Default Configuration
# ============================================================================

# Model configuration
MODEL_PATH="${MODEL_PATH:-<your-org>/Explore-Execute-Chain}"
SUBFOLDER="${SUBFOLDER:-4B-Final}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"  # Optional: load weights from checkpoint

# Evaluation settings
DATASET="${DATASET:-gsm8k}"  # Options: gsm8k, math, aime24, aime25, amc23, all, med
SAMPLE_NUM="${SAMPLE_NUM:-1}"  # Number of samples per question
BATCH_SIZE="${BATCH_SIZE:--1}"  # -1 for auto
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:--1}"  # -1 for auto
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

# System settings
N_GPUS="${N_GPUS:-1}"
SEED="${SEED:-0}"
SAVE_PATH="${SAVE_PATH:-evaluation/e2c-eval}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --subfolder)
            SUBFOLDER="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --sample)
            SAMPLE_NUM="$2"
            shift 2
            ;;
        --gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --temp)
            TEMPERATURE="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Display Configuration
# ============================================================================

echo "=========================================="
echo "E2C Complete Evaluation Pipeline"
echo "=========================================="
echo "Model Path:     $MODEL_PATH"
if [ -n "$SUBFOLDER" ]; then
    echo "Subfolder:      $SUBFOLDER"
fi
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint:     $CHECKPOINT_PATH"
fi
echo "Dataset(s):     $DATASET"
echo "Sample Num:     $SAMPLE_NUM"
echo "Temperature:    $TEMPERATURE"
echo "GPUs:           $N_GPUS"
echo "Master Port:    $MASTER_PORT"
echo "Save Path:      $SAVE_PATH"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$SAVE_PATH"

# ============================================================================
# Prepare model path override
# ============================================================================

if [ -n "$SUBFOLDER" ]; then
    MODEL_ARG="model.model_path=$MODEL_PATH model.subfolder=$SUBFOLDER"
else
    MODEL_ARG="model.model_path=$MODEL_PATH"
fi

# Add checkpoint if specified
if [ -n "$CHECKPOINT_PATH" ]; then
    MODEL_ARG="$MODEL_ARG model.checkpoint_path=$CHECKPOINT_PATH"
fi

# ============================================================================
# Stage 1: Generate model outputs
# ============================================================================

echo "=========================================="
echo "Stage 1/2: Generating Model Outputs"
echo "=========================================="
echo ""

torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=$MASTER_PORT \
    e2c/inference/generate.py \
    --config-path="../config" \
    --config-name="eval" \
    $MODEL_ARG \
    eval.dataset="['$DATASET']" \
    eval.sample_num=$SAMPLE_NUM \
    eval.batch_size=$BATCH_SIZE \
    eval.max_new_tokens=$MAX_NEW_TOKENS \
    eval.temperature=$TEMPERATURE \
    eval.top_p=$TOP_P \
    eval.seed=$SEED \
    eval.save_path="$SAVE_PATH" \
    eval.backend=hf

GENERATE_EXIT_CODE=$?

if [ $GENERATE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "❌ Generation failed with exit code $GENERATE_EXIT_CODE"
    echo "=========================================="
    exit $GENERATE_EXIT_CODE
fi

echo ""
echo "✅ Generation completed successfully!"
echo ""

# ============================================================================
# Stage 2: Evaluate generated outputs
# ============================================================================

echo "=========================================="
echo "Stage 2/2: Evaluating Generated Outputs"
echo "=========================================="
echo ""

torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=$MASTER_PORT \
    e2c/inference/evaluate.py \
    --config-path="../config" \
    --config-name="eval" \
    $MODEL_ARG \
    eval.dataset="['$DATASET']" \
    eval.sample_num=$SAMPLE_NUM \
    eval.batch_size=$BATCH_SIZE \
    eval.max_new_tokens=$MAX_NEW_TOKENS \
    eval.temperature=$TEMPERATURE \
    eval.top_p=$TOP_P \
    eval.seed=$SEED \
    eval.save_path="$SAVE_PATH" \
    eval.backend=hf

EVAL_EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Complete Evaluation Pipeline Finished!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $SAVE_PATH"
    echo ""
    echo "📊 View results:"
    echo "   - Detailed results: ${SAVE_PATH}/${DATASET}/result_${SEED}_merged.json"
    echo "   - Statistics: ${SAVE_PATH}/${DATASET}/static_${SEED}_merged.json"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results:"
    echo "     cat ${SAVE_PATH}/${DATASET}/static_${SEED}_merged.json"
    echo ""
    echo "  2. Evaluate on more datasets:"
    echo "     bash e2c/inference/run_eval.sh --dataset math --sample 4"
    echo ""
    echo "  3. Evaluate all math benchmarks:"
    echo "     bash e2c/inference/run_eval.sh --dataset all --sample 8"
else
    echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "=========================================="
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "Done! 🎉"

