#!/bin/bash
#
# E2C Model Generation Script
# Generate responses from trained E2C models
# Can be run from scripts directory
#
# Usage:
#   1. Basic generation (GSM8K with VLLM):
#      bash generate.sh
#
#   2. Generate on specific dataset:
#      bash generate.sh --dataset math
#
#   3. Generate with custom model:
#      bash generate.sh --model /path/to/your/model
#
#   4. Generate with traditional model (uses torchrun):
#      bash generate.sh --model-type traditional --gpus 2
#
#   5. Generate on all math benchmarks:
#      bash generate.sh --dataset all --sample 4
#
#   6. Generate with multiple samples:
#      bash generate.sh --dataset gsm8k --sample 8 --temp 0.7
#
#   7. Generate with VLLM (explicit):
#      bash generate.sh --model-type vllm --dataset gsm8k

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

# Model configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"  # Optional: load weights from checkpoint
MODEL_TYPE="${MODEL_TYPE:-vllm}"  # Options: vllm, traditional

# Generation settings
DATASET="${DATASET:-gsm8k}"  # Options: gsm8k, math, aime24, aime25, amc23, all, med
SAMPLE_NUM="${SAMPLE_NUM:-1}"  # Number of samples per question
BATCH_SIZE="${BATCH_SIZE:--1}"  # -1 for auto
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:--1}"  # -1 for auto
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

# System settings
N_GPUS="${N_GPUS:-1}"
SEED="${SEED:-0}"
SAVE_PATH="${SAVE_PATH:-./generations}"

# SPECIAL settings
SAVE_AS_DATASET="${SAVE_AS_DATASET:-False}"
RESUME_DIR="${RESUME_DIR:-}"



# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
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
        --save-as-dataset)
            SAVE_AS_DATASET="$2"
            shift 2
            ;;
        --resume-dir)
            RESUME_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "E2C Generation Script"
            echo ""
            echo "Usage: bash generate.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH           Model path (default: Qwen/Qwen3-8B)"
            echo "  --model-type TYPE      Model type: vllm or traditional (default: vllm)"
            echo "  --checkpoint PATH      Checkpoint path (optional)"
            echo "  --dataset DATASET      Dataset: gsm8k, math, aime24, aime25, amc23, all, med (default: gsm8k)"
            echo "  --sample NUM           Number of samples per question (default: 1)"
            echo "  --gpus NUM             Number of GPUs (default: 1)"
            echo "  --temp TEMP            Temperature (default: 1.0)"
            echo "  --save-path PATH       Save path (default: ./generations)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash generate.sh                                    # Basic VLLM generation"
            echo "  bash generate.sh --dataset math --sample 4         # Math dataset, 4 samples"
            echo "  bash generate.sh --model-type traditional --gpus 2 # Traditional model with 2 GPUs"
            echo "  bash generate.sh --model /path/to/model            # Custom model path"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Display Configuration
# ============================================================================

echo "=========================================="
echo "E2C Generation Configuration"
echo "=========================================="
echo "Model Path:     $MODEL_PATH"
echo "Model Type:     $MODEL_TYPE"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint:     $CHECKPOINT_PATH"
fi
echo "Dataset(s):     $DATASET"
echo "Sample Num:     $SAMPLE_NUM"
echo "Temperature:    $TEMPERATURE"
echo "GPUs:           $N_GPUS"
echo "Save Path:      $SAVE_PATH"
echo "Save As Dataset: $SAVE_AS_DATASET"
echo "Resume Dir:      $RESUME_DIR"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$SAVE_PATH"

# ============================================================================
# Run Generation
# ============================================================================

echo "Starting generation..."
echo ""

# Prepare model path override (escape spaces properly)
MODEL_ARG="model.model_path=\"$MODEL_PATH\" model.type=$MODEL_TYPE"

# Add checkpoint if specified
if [ -n "$CHECKPOINT_PATH" ]; then
    MODEL_ARG="$MODEL_ARG model.checkpoint_path=\"$CHECKPOINT_PATH\""
fi

# Set Python path to include e2c directory for module imports
export PYTHONPATH="${PROJECT_ROOT}/e2c:${PYTHONPATH}"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'verl'..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate verl
fi

# Choose execution method based on model type
if [ "$MODEL_TYPE" = "traditional" ]; then
    echo "🚀 Running with torchrun (distributed training)..."
    echo "   Using $N_GPUS GPUs for traditional model"
    
    # Run generation with torchrun (distributed)
    torchrun \
        --nproc_per_node=$N_GPUS \
        --master_port=29500 \
        e2c/inference/generate.py \
        --config-path="../config" \
        --config-name="generate" \
        $MODEL_ARG \
        generation.dataset="['$DATASET']" \
        generation.sample_num=$SAMPLE_NUM \
        generation.batch_size=$BATCH_SIZE \
        generation.max_new_tokens=$MAX_NEW_TOKENS \
        generation.temperature=$TEMPERATURE \
        generation.top_p=$TOP_P \
        generation.seed=$SEED \
        generation.save_path="$SAVE_PATH" \
        generation.save_as_dataset=$SAVE_AS_DATASET \
        generation.resume_dir="$RESUME_DIR"
else
    echo "🚀 Running with VLLM (single process, multi-GPU)..."
    echo "   VLLM will automatically use all available GPUs"
    
    # Run generation with python3 (VLLM handles multi-GPU internally)
    python3 e2c/inference/generate.py \
        --config-path="../config" \
        --config-name="generate" \
        $MODEL_ARG \
        generation.dataset="['$DATASET']" \
        generation.sample_num=$SAMPLE_NUM \
        generation.batch_size=$BATCH_SIZE \
        generation.max_new_tokens=$MAX_NEW_TOKENS \
        generation.temperature=$TEMPERATURE \
        generation.top_p=$TOP_P \
        generation.seed=$SEED \
        generation.save_path="$SAVE_PATH" \
        generation.save_as_dataset=$SAVE_AS_DATASET \
        generation.resume_dir="$RESUME_DIR"
fi

GEN_EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $GEN_EXIT_CODE -eq 0 ]; then
    echo "✅ Generation Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $SAVE_PATH"
    echo ""
    echo "📊 Generated files:"
    echo "   - Generations: ${SAVE_PATH}/${DATASET}/generations_${SEED}_merged.json"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate the generations:"
    echo "     bash scripts/eval_only.sh --generation-path $SAVE_PATH --dataset $DATASET"
    echo ""
    echo "  2. Generate on more datasets:"
    echo "     bash scripts/generate.sh --dataset math --sample 4"
    echo ""
    echo "  3. Generate all math benchmarks:"
    echo "     bash scripts/generate.sh --dataset all --sample 8"
else
    echo "❌ Generation failed with exit code $GEN_EXIT_CODE"
    echo "=========================================="
    exit $GEN_EXIT_CODE
fi

echo ""
echo "Done! 🎉"

