#!/bin/bash
#
# E2C-RL Training Script
# This script fine-tunes the E2C-SFT model using reinforcement learning (RL)
# to improve reasoning quality through policy optimization.
# Can be run from scripts directory
#
# The script supports two-stage RL training:
#   Stage 1: Warm-up with larger rollout samples
#   Stage 2: Fine-tuning with constrained rewards
#
# Usage:
#   1. Run from scripts directory (both stages):
#      bash e2c_rl.sh
#
#   2. Or customize with environment variables:
#      export MODEL_PATH="models/checkpoints/sft/final"
#      bash e2c_rl.sh
#
#   3. Or run specific stage only:
#      bash e2c_rl.sh --stage 2    # Only run stage 2
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
OUTPUT_DIR="${OUTPUT_DIR:-models/checkpoints/rl}"
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
# These are token IDs for "<EXECUTION>" markers
SPECIAL_TOKEN_1="${SPECIAL_TOKEN_1:-151672}"
SPECIAL_TOKEN_2="${SPECIAL_TOKEN_2:-151673}"

# Common hyperparameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-712}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PROJECT_NAME="${PROJECT_NAME:-e2c-rl}"

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
echo "E2C-RL Training Configuration"
echo "=========================================="
echo "Model Path:        $MODEL_PATH"
echo "Training Data:     $TRAIN_DATA"
echo "Validation Data:   $VAL_DATA"
echo "Output Directory:  $OUTPUT_DIR"
echo "GPUs:              $N_GPUS"
echo "Run Stage 1:       $([ $RUN_STAGE1 -eq 1 ] && echo 'Yes' || echo 'No')"
echo "Run Stage 2:       $([ $RUN_STAGE2 -eq 1 ] && echo 'Yes' || echo 'No')"
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
    echo "Starting Stage 1: Warm-up Training"
    echo "=========================================="
    echo "Parameters:"
    echo "  - Rollout samples: 32"
    echo "  - Epochs: 1"
    echo "  - Advantage coeff: 1.0"
    echo "  - Temperature: 1.3"
    echo "  - Constrained reward: False"
    echo "=========================================="
    echo ""

    STAGE1_TEMPERATURE=1.3
    STAGE1_ADV_COEFF=1.0
    STAGE1_ROLLOUT=32
    STAGE1_EPOCHS=1
    STAGE1_USE_CONSTRAIN=False

# Add verl to PYTHONPATH for Ray workers
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH}"

# Also ensure Ray workers can find the module
cd "${PROJECT_ROOT}/verl"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_DATA" \
        data.val_files="$VAL_DATA" \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.question_suffix="" \
    data.solution_prefix="'<EXPLORATION>'" \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.fixed_exploration_rate=0 \
        data.flexible_instruction=0 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
        reward_model.use_constrain_reward=$STAGE1_USE_CONSTRAIN \
        reward_model.reward_manager=dapo \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=14000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.adv_coeff=$STAGE1_ADV_COEFF \
    actor_rollout_ref.actor.log_part_entropy=True \
        actor_rollout_ref.actor.special_token_1=$SPECIAL_TOKEN_1 \
        actor_rollout_ref.actor.special_token_2=$SPECIAL_TOKEN_2 \
        actor_rollout_ref.actor.entropy_mask_coef_after=1.0 \
        actor_rollout_ref.actor.entropy_mask_coef_before=1.0 \
        actor_rollout_ref.actor.entropy_mask_min_position=5 \
        actor_rollout_ref.actor.kl_mask_coef_after=1.0 \
        actor_rollout_ref.actor.kl_mask_coef_before=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=$STAGE1_TEMPERATURE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.n=$STAGE1_ROLLOUT \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="stage1-warmup" \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=10 \
        trainer.total_epochs=$STAGE1_EPOCHS \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
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
# Stage 2: Main RL Training (Recommended)
# ============================================================================

if [ $RUN_STAGE2 -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo "Starting Stage 2: Main RL Training"
    echo "=========================================="
    echo "Parameters:"
    echo "  - Rollout samples: 8"
    echo "  - Epochs: 2"
    echo "  - Advantage coeff: 2.0"
    echo "  - Temperature: 1.0"
    echo "  - Constrained reward: True ✓"
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
cd "${PROJECT_ROOT}/verl"

    set -x

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_DATA" \
        data.val_files="$VAL_DATA" \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.question_suffix="" \
        data.solution_prefix="'<EXPLORATION>'" \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.fixed_exploration_rate=0 \
        data.flexible_instruction=0 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        reward_model.use_constrain_reward=$STAGE2_USE_CONSTRAIN \
        reward_model.reward_manager=dapo \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=14000 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.adv_coeff=$STAGE2_ADV_COEFF \
        actor_rollout_ref.actor.log_part_entropy=True \
        actor_rollout_ref.actor.special_token_1=$SPECIAL_TOKEN_1 \
        actor_rollout_ref.actor.special_token_2=$SPECIAL_TOKEN_2 \
        actor_rollout_ref.actor.entropy_mask_coef_after=1.0 \
        actor_rollout_ref.actor.entropy_mask_coef_before=1.0 \
        actor_rollout_ref.actor.entropy_mask_min_position=5 \
        actor_rollout_ref.actor.kl_mask_coef_after=1.0 \
        actor_rollout_ref.actor.kl_mask_coef_before=1.0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=$STAGE2_TEMPERATURE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.n=$STAGE2_ROLLOUT \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="stage2-main" \
        trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
        trainer.save_freq=20 \
    trainer.test_freq=10 \
        trainer.total_epochs=$STAGE2_EPOCHS \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
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
echo "✓ E2C-RL Training Complete!"
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

