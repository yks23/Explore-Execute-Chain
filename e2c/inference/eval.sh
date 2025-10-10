# Here is an example for eval.sh
# You can modify it according to your needs
torchrun --nproc_per_node=1 inference/eval.py --config-path config --config-name eval.yaml