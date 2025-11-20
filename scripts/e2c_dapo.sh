#!/bin/bash
#
# E2C-DAPO Training Script
# This script fine-tunes the E2C-SFT model using DAPO (Direct Alignment from Preference Optimization)
# with token-level operations for exploration-execution chain alignment.
# Can be run from scripts directory
#
# The script supports two-stage DAPO training:
#   Stage 1: Warm-up with larger rollout samples
#   Stage 2: Fine-tuning with constrained rewards
#
# Usage:
#   1. Run from scripts directory (both stages):
#      bash e2c_dapo.sh
#
#   2. Or customize with environment variables:
#      export MODEL_PATH="models/checkpoints/sft/final"
#      bash e2c_dapo.sh
#
#   3. Or run specific stage only:
#      bash e2c_dapo.sh --stage 2    # Only run stage 2
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

# Model paths (should be SFT checkpoint)
MODEL_PATH="${MODEL_PATH:-models/checkpoints/sft/final}"

# Data paths - convert to absolute paths
TRAIN_DATA="${TRAIN_DATA:-data/processed/rl/e2c-rl-train.parquet}"
VAL_DATA="${VAL_DATA:-data/processed/rl/e2c-rl-val.parquet}"

# Convert relative paths to absolute paths
if [[ "$TRAIN_DATA" != /* ]]; then
    TRAIN_DATA="${PROJECT_ROOT}/${TRAIN_DATA}"
fi
if [[ "$VAL_DATA" != /* ]]; then
    VAL_DATA="${PROJECT_ROOT}/${VAL_DATA}"
fi
# Convert MODEL_PATH only if it's a relative local path (not HF model ID)
if [[ "$MODEL_PATH" != /* ]] && [[ "$MODEL_PATH" == models/* || "$MODEL_PATH" == ./* || "$MODEL_PATH" == ../* ]]; then
    MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH}"
fi

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-models/checkpoints/dapo}"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

# Training stages (set to 0 to skip, 1 to run)
# Default: Run both stages for complete training
RUN_STAGE1="${RUN_STAGE1:-1}"  # Stage 1: Warm-up training
RUN_STAGE2="${RUN_STAGE2:-1}"  # Stage 2: Main training

# GPU configuration
N_GPUS="${N_GPUS:-8}"

# Special tokens for Qwen3 model (for exploration/execution split)
# These are token IDs for "</EXPLORATION>" and "<EXECUTION>" markers
SPECIAL_TOKEN_1="${SPECIAL_TOKEN_1:-151672}"  # </EXPLORATION>
SPECIAL_TOKEN_2="${SPECIAL_TOKEN_2:-151673}"  # <EXECUTION>

# Common hyperparameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"  # 2 * 1024
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"  # 16 * 1024
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PROJECT_NAME="${PROJECT_NAME:-e2c-dapo}"

# DAPO specific hyperparameters (based on run_dapo_qwen2.5_32b.sh)
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
ENABLE_FILTER_GROUPS="${ENABLE_FILTER_GROUPS:-True}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 3))}"  # gen_prompt_bsz = train_prompt_bsz * 3

# Optimizer settings
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Sampling parameters
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"  # -1 for vLLM rollout
VAL_TOP_P="${VAL_TOP_P:-0.7}"

# Overlong buffer settings
ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-True}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}"  # 4 * 1024
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

# Performance settings
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
OFFLOAD="${OFFLOAD:-True}"
GEN_TP="${GEN_TP:-1}"  # tensor parallel size for generation
SP_SIZE="${SP_SIZE:-1}"  # sequence parallel size (ulysses)

# vLLM settings
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            if [ "$2" = "1" ]; then
                RUN_STAGE1=1
                RUN_STAGE2=0
            elif [ "$2" = "2" ]; then
                RUN_STAGE1=0
                RUN_STAGE2=1
            fi
            shift 2
            ;;
        --both)
            RUN_STAGE1=1
            RUN_STAGE2=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

# ============================================================================
# Validate Configuration
# ============================================================================

echo "=========================================="
echo "E2C-DAPO Training Configuration"
echo "=========================================="
echo "Model Path:        $MODEL_PATH"
echo "Training Data:     $TRAIN_DATA"
echo "Validation Data:   $VAL_DATA"
echo "Output Directory:  $OUTPUT_DIR"
echo "GPUs:              $N_GPUS"
echo "Run Stage 1:       $([ $RUN_STAGE1 -eq 1 ] && echo 'Yes' || echo 'No')"
echo "Run Stage 2:       $([ $RUN_STAGE2 -eq 1 ] && echo 'Yes' || echo 'No')"
echo ""
echo "Token Operations:"
echo "  Special Token 1:   $SPECIAL_TOKEN_1 (</EXPLORATION>)"
echo "  Special Token 2:   $SPECIAL_TOKEN_2 (<EXECUTION>)"
echo ""
echo "DAPO Hyperparameters:"
echo "  Clip Ratio:        [$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH] (c=$CLIP_RATIO_C)"
echo "  Loss Agg Mode:     $LOSS_AGG_MODE"
echo "  Filter Groups:     $ENABLE_FILTER_GROUPS (metric=$FILTER_GROUPS_METRIC)"
echo "  Max Gen Batches:   $MAX_NUM_GEN_BATCHES"
echo "  Gen Batch Size:    $GEN_BATCH_SIZE (train=$TRAIN_BATCH_SIZE)"
echo ""
echo "Optimizer:"
echo "  Learning Rate:     $LEARNING_RATE"
echo "  Warmup Steps:     $LR_WARMUP_STEPS"
echo "  Weight Decay:     $WEIGHT_DECAY"
echo "  Grad Clip:        $GRAD_CLIP"
echo ""
echo "Sampling:"
echo "  Temperature:      $TEMPERATURE"
echo "  Top-p:            $TOP_P"
echo "  Top-k:            $TOP_K"
echo "  Val Top-p:        $VAL_TOP_P"
echo ""
echo "Performance:"
echo "  Max Prompt Len:   $MAX_PROMPT_LENGTH"
echo "  Max Response Len: $MAX_RESPONSE_LENGTH"
echo "  Overlong Buffer:  $ENABLE_OVERLONG_BUFFER (len=$OVERLONG_BUFFER_LEN)"
echo "  GPU Memory Util:  $GPU_MEMORY_UTILIZATION"
echo "  Gen TP Size:      $GEN_TP"
echo "  Sequence Parallel: $SP_SIZE"
echo "=========================================="
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
# Stage 1: Warm-up Training (Optional)
# ============================================================================

if [ $RUN_STAGE1 -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo "Starting Stage 1: Warm-up Training (DAPO)"
    echo "=========================================="
    echo "Parameters:"
    echo "  - Rollout samples: 32"
    echo "  - Epochs: 1"
    echo "  - Advantage coeff: 1.0"
    echo "  - Temperature: 1.3"
    echo "  - Constrained reward: False"
    echo "  - Token operations: Enabled"
    echo "=========================================="
    echo ""

    STAGE1_TEMPERATURE=1.3
    STAGE1_ADV_COEFF=1.0
    STAGE1_ROLLOUT=32
    STAGE1_EPOCHS=1
    STAGE1_USE_CONSTRAIN=False

    # Add verl to PYTHONPATH for Ray workers
    export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH}"

    # Ensure Ray workers can find the module
    cd "${PROJECT_ROOT}/verl/recipe/dapo"

    set -x

    python3 -m verl.recipe.dapo.main_dapo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.filter_groups.enable=$ENABLE_FILTER_GROUPS \
        algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES \
        algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC \
        data.train_files="$TRAIN_DATA" \
        data.val_files="$VAL_DATA" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.gen_batch_size=$GEN_BATCH_SIZE \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.question_suffix="" \
        data.solution_prefix="'<EXPLORATION>'" \
        actor_rollout_ref.rollout.n=$STAGE1_ROLLOUT \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$INFER_PPO_MAX_TOKEN_LEN \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$INFER_PPO_MAX_TOKEN_LEN \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
        actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
        actor_rollout_ref.actor.clip_ratio_c=$CLIP_RATIO_C \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
        actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
        actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
        actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
        actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
        actor_rollout_ref.actor.adv_coeff=$STAGE1_ADV_COEFF \
        actor_rollout_ref.actor.log_part_entropy=True \
        actor_rollout_ref.actor.special_token_1=$SPECIAL_TOKEN_1 \
        actor_rollout_ref.actor.special_token_2=$SPECIAL_TOKEN_2 \
        actor_rollout_ref.actor.entropy_mask_coef_after=1.0 \
        actor_rollout_ref.actor.entropy_mask_coef_before=1.0 \
        actor_rollout_ref.actor.entropy_mask_min_position=5 \
        actor_rollout_ref.actor.kl_mask_coef_after=1.0 \
        actor_rollout_ref.actor.kl_mask_coef_before=1.0 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
        actor_rollout_ref.rollout.enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL \
        actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
        actor_rollout_ref.rollout.temperature=$TEMPERATURE \
        actor_rollout_ref.rollout.top_p=$TOP_P \
        actor_rollout_ref.rollout.top_k="$TOP_K" \
        actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
        actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
        actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE \
        reward_model.reward_manager=dapo \
        reward_model.use_constrain_reward=$STAGE1_USE_CONSTRAIN \
        reward_model.overlong_buffer.enable=$ENABLE_OVERLONG_BUFFER \
        reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
        reward_model.overlong_buffer.penalty_factor=$OVERLONG_PENALTY_FACTOR \
        trainer.logger='["console"]' \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="stage1-warmup" \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.test_freq=10 \
        trainer.save_freq=20 \
        trainer.total_epochs=$STAGE1_EPOCHS \
        trainer.default_local_dir="$OUTPUT_DIR" \
        trainer.resume_mode=auto \
        $@

    STAGE1_EXIT_CODE=$?
    set +x

    if [ $STAGE1_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "❌ Stage 1 training failed."
        exit 1
    fi

    echo ""
    echo "✓ Stage 1 training complete!"
    echo ""

    # Update model path for stage 2 if running both stages
    if [ $RUN_STAGE2 -eq 1 ]; then
        echo ""
        echo "Looking for Stage 1 checkpoint to use for Stage 2..."
        # Find the latest checkpoint from stage 1 - look for final checkpoint
        if [ -d "$OUTPUT_DIR/stage1-warmup" ]; then
            # Use the final checkpoint from stage 1
            MODEL_PATH="$OUTPUT_DIR/stage1-warmup"
            echo "✓ Stage 2 will use checkpoint from Stage 1: $MODEL_PATH"
        else
            echo "⚠️  Stage 1 checkpoint not found at expected location."
            echo "   Stage 2 will use original MODEL_PATH: $MODEL_PATH"
        fi
        echo ""
    fi
fi

# ============================================================================
# Stage 2: Main DAPO Training (Recommended)
# ============================================================================

if [ $RUN_STAGE2 -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo "Starting Stage 2: Main DAPO Training"
    echo "=========================================="
    echo "Parameters:"
    echo "  - Rollout samples: 8"
    echo "  - Epochs: 2"
    echo "  - Advantage coeff: 2.0"
    echo "  - Temperature: 1.0"
    echo "  - Constrained reward: True ✓"
    echo "  - Token operations: Enabled"
    echo "=========================================="
    echo ""

    STAGE2_TEMPERATURE=1.0
    STAGE2_ADV_COEFF=2.0
    STAGE2_ROLLOUT=8
    STAGE2_EPOCHS=2
    STAGE2_USE_CONSTRAIN=True

    # Add verl to PYTHONPATH (in case stage 2 runs independently)
    export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH}"

    # Ensure we're in the verl directory for Ray workers
    cd "${PROJECT_ROOT}/verl/recipe/dapo"

    set -x

    python3 -m verl.recipe.dapo.main_dapo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.filter_groups.enable=$ENABLE_FILTER_GROUPS \
        algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES \
        algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC \
        data.train_files="$TRAIN_DATA" \
        data.val_files="$VAL_DATA" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.gen_batch_size=$GEN_BATCH_SIZE \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.question_suffix="" \
        data.solution_prefix="'<EXPLORATION>'" \
        actor_rollout_ref.rollout.n=$STAGE2_ROLLOUT \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$INFER_PPO_MAX_TOKEN_LEN \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$INFER_PPO_MAX_TOKEN_LEN \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
        actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
        actor_rollout_ref.actor.clip_ratio_c=$CLIP_RATIO_C \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
        actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
        actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
        actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
        actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
        actor_rollout_ref.actor.adv_coeff=$STAGE2_ADV_COEFF \
        actor_rollout_ref.actor.log_part_entropy=True \
        actor_rollout_ref.actor.special_token_1=$SPECIAL_TOKEN_1 \
        actor_rollout_ref.actor.special_token_2=$SPECIAL_TOKEN_2 \
        actor_rollout_ref.actor.entropy_mask_coef_after=1.0 \
        actor_rollout_ref.actor.entropy_mask_coef_before=1.0 \
        actor_rollout_ref.actor.entropy_mask_min_position=5 \
        actor_rollout_ref.actor.kl_mask_coef_after=1.0 \
        actor_rollout_ref.actor.kl_mask_coef_before=1.0 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
        actor_rollout_ref.rollout.enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL \
        actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
        actor_rollout_ref.rollout.temperature=$TEMPERATURE \
        actor_rollout_ref.rollout.top_p=$TOP_P \
        actor_rollout_ref.rollout.top_k="$TOP_K" \
        actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
        actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
        actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE \
        reward_model.reward_manager=dapo \
        reward_model.use_constrain_reward=$STAGE2_USE_CONSTRAIN \
        reward_model.overlong_buffer.enable=$ENABLE_OVERLONG_BUFFER \
        reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
        reward_model.overlong_buffer.penalty_factor=$OVERLONG_PENALTY_FACTOR \
        trainer.logger='["console"]' \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="stage2-main" \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.test_freq=10 \
        trainer.save_freq=20 \
        trainer.total_epochs=$STAGE2_EPOCHS \
        trainer.default_local_dir="$OUTPUT_DIR" \
        trainer.resume_mode=auto \
        $@

    STAGE2_EXIT_CODE=$?
    set +x

    if [ $STAGE2_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "❌ Stage 2 training failed."
        exit 1
    fi

    echo ""
    echo "✓ Stage 2 training complete!"
    echo ""
fi

# ============================================================================
# Training Complete
# ============================================================================

echo ""
echo "=========================================="
echo "✓ E2C-DAPO Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Training Summary:"
if [ $RUN_STAGE1 -eq 1 ]; then
    echo "  ✓ Stage 1 (Warm-up): Complete"
fi
if [ $RUN_STAGE2 -eq 1 ]; then
    echo "  ✓ Stage 2 (Main): Complete"
fi
echo ""
echo "Next steps:"
echo "  1. Find your checkpoints:"
echo "     ls -lh $OUTPUT_DIR"
echo ""
echo "  2. Run evaluation:"
echo "     export MODEL_PATH=\"$OUTPUT_DIR/stage2-main/final\""
echo "     bash scripts/eval.sh"
echo ""
echo "  3. Or adapt to new domain (EF-SFT):"
echo "     export MODEL_PATH=\"$OUTPUT_DIR/stage2-main/final\""
echo "     bash scripts/ef-sft.sh"
echo ""
echo "  4. Or run interactive demo:"
echo "     python example_interactive.py --model-path \"$OUTPUT_DIR/stage2-main/final\""
echo "=========================================="

