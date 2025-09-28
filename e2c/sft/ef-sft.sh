export train_file=./data/ef-sft-train.parquet
export val_file=./data/ef-sft-val.parquet
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_NUM=4
export MASTER_PORT=12345
export MODEL_PATH="Path to the pretrained model"
export base_dir=""
# 500 iter is enough
export total_training_steps=500
torchrun --nproc_per_node $CUDA_NUM ".\verl\verl\trainer\fsdp_sft_trainer.py" --config-path config --config-name sft_trainer.yaml \
  model.partial_pretrain=$MODEL_PATH \
  data.train_files=$train_file \
  data.val_files=$val_file \
  trainer.project_name=e2c-sft \
  trainer.experiment_name=run1 \
  trainer.base_dir=$base_dir \
  trainer.total_training_steps=$total_training_steps \
  trainer.n_gpus_per_node=$CUDA_NUM \
