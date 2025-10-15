#!/bin/bash
#
# E2C Evaluation-Only Script
# Evaluate pre-generated responses
# Can be run from scripts directory
#
# Usage:
#   1. Basic evaluation (GSM8K):
#      bash eval_only.sh --generation-path ./generations
#
#   2. Evaluate specific dataset:
#      bash eval_only.sh --generation-path ./generations --dataset math
#
#   3. Evaluate all math benchmarks:
#      bash eval_only.sh --generation-path ./generations --dataset all
#
#   4. Custom save path:
#      bash eval_only.sh --generation-path ./generations --save-path evaluation/my-eval

# ============================================================================
# Get script directory and project root
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# ============================================================================
# Default Configuration
# ============================================================================

# Generation path (where generations are stored)
GENERATION_PATH="${GENERATION_PATH:-./generations}"

# Evaluation settings
DATASET="${DATASET:-gsm8k}"  # Options: gsm8k, math, aime24, aime25, amc23, all, med
SEED="${SEED:-0}"
SAVE_PATH="${SAVE_PATH:-evaluation/e2c-eval}"

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --generation-path)
            GENERATION_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
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
echo "E2C Evaluation-Only Configuration"
echo "=========================================="
echo "Generation Path: $GENERATION_PATH"
echo "Dataset(s):      $DATASET"
echo "Seed:            $SEED"
echo "Save Path:       $SAVE_PATH"
echo "=========================================="
echo ""

# Check if generation path exists
if [ ! -d "$GENERATION_PATH" ]; then
    echo "❌ Error: Generation path does not exist: $GENERATION_PATH"
    echo "Please run generation first:"
    echo "  bash scripts/generate.sh --dataset $DATASET"
    exit 1
fi

# Create output directory
mkdir -p "$SAVE_PATH"

# ============================================================================
# Run Evaluation
# ============================================================================

echo "Starting evaluation..."
echo ""

# Set Python path to include e2c directory for module imports
export PYTHONPATH="${PROJECT_ROOT}/e2c:${PYTHONPATH}"

# Run evaluation (simplified - no distributed processing)
python e2c/inference/eval_only.py \
    --config-path="../config" \
    --config-name="eval_only" \
    eval.dataset="['$DATASET']" \
    eval.generation_path="$GENERATION_PATH" \
    eval.seed=$SEED \
    eval.save_path="$SAVE_PATH"

EVAL_EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $SAVE_PATH"
    echo ""
    echo "📊 View results:"
    echo "   - Detailed results: ${SAVE_PATH}/${DATASET}/result_${SEED}_merged.json"
    echo "   - Statistics: ${SAVE_PATH}/${DATASET}/static_${SEED}_merged.json"
    echo ""
    echo "View statistics:"
    echo "   cat ${SAVE_PATH}/${DATASET}/static_${SEED}_merged.json"
    echo ""
else
    echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "=========================================="
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "Done! 🎉"

