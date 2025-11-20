#!/bin/bash

# This script sets up a development environment using Tsinghua mirrors for faster downloads.

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    # Added Tsinghua mirror for sglang and torch-memory-saver
    pip install "sglang[all]==0.4.6.post1" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && pip install torch-memory-saver --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
# Added Tsinghua mirror for vllm, torch, and other related packages
pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "2. install basic packages"
# Added Tsinghua mirror for all basic packages
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

# Added Tsinghua mirror for fastapi and other network-related packages
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" -i https://pypi.tuna.tsinghua.edu.cn/simple


echo "3. install FlashAttention and FlashInfer"
# The wget commands for these are specific URLs, so no mirror change is needed.
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    # Added Tsinghua mirror for TransformerEngine and Megatron-LM from their git repos
    # Note: These are git installs, so the mirror may not apply to the git clone part,
    # but it will handle dependencies from PyPI.
    NVTE_FRAMEWORK=pytorch pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
fi


echo "5. May need to fix opencv"
# Added Tsinghua mirror for opencv packages
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-fixer -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    # Added Tsinghua mirror for cudnn package
    pip install nvidia-cudnn-cu12==9.8.0.87 -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo "Successfully installed all packages"