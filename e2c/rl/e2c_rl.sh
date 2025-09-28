# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
# ray start --head --port=6379 --num-gpus=4
ray start --head --port=6379 --num-gpus=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_NUM=8
MASTER_PORT=12345
MODEL_PATH="Path to the pretrained model"
base_dir="./training"
total_epochs=2
train_file=./data/e2c-rl-train.parquet
val_file=./data/e2c-rl-val.parquet
adv_coeff=2.0
project_name=E2C-RL
experiment_name=run1
set -x
# ROLL_TEMP=1.3
python3 -m verl.trainer.main_ppo \
    --config-path ./config --config-name ppo_trainer.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files="/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/dapo-diverse.parquet" \
    data.val_files="/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/qwen-rl-valid.parquet" \
    data.train_batch_size=256 \
    data.question_suffix="" \
    data.solution_prefix="'<EXPLORATION>'" \
    data.max_prompt_length=712 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    reward_model.use_constrain_reward=True \
    actor_rollout_ref.model.path="/WORK/fit/alex/Kaisen/checkpoints/qwen/combine/global_step_500/huggingface" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=14000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.adv_coeff=$adv_coeff \
    actor_rollout_ref.actor.log_part_entropy=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=8192 \
    trainer.total_epochs=3 $@

# change rl_qwen
# change naive for decoding
# change rl_dataset for loading


# temp sample epoch fix