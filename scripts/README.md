# E2C Scripts 目录

此目录包含所有用于训练、评估和数据准备的可执行脚本。所有脚本都可以直接在此目录下运行。

## 📝 脚本列表

### 数据准备

#### `prepare_all_data.sh`
一键准备所有训练数据（SFT + RL）

```bash
bash prepare_all_data.sh
```

**选项：**
- `--skip-download` - 跳过数据下载
- `--skip-sft` - 跳过 SFT 数据准备
- `--skip-rl` - 跳过 RL 数据准备
- `--num-workers N` - 并行工作进程数（默认: 8）

#### `download_datasets.sh`
下载原始数据集

```bash
# 基础用法
bash download_datasets.sh            # 下载所有数据
bash download_datasets.sh --dataset sft    # 仅下载 SFT 数据
bash download_datasets.sh --dataset rl     # 仅下载 RL 数据
bash download_datasets.sh --dataset eval   # 仅下载评估数据（16个数据集）

# 使用镜像源（国内用户推荐，速度更快）
bash download_datasets.sh --mirror         # 使用镜像源下载所有数据
bash download_datasets.sh --dataset eval --mirror  # 使用镜像源下载评估数据

# 通过环境变量设置镜像源
export HF_ENDPOINT=https://hf-mirror.com
bash download_datasets.sh
```

**数据文件说明：**
- **SFT 数据**: `e2c-sft.parquet` (77.7 MB) → 保存到 `data/raw/sft/`
- **RL 数据**: 
  - `e2c-rl.parquet` (19.4 MB) → 保存到 `data/raw/rl/`
  - `e2c-rl-valid.parquet` (706 KB) → 保存到 `data/raw/rl/`
  - 注：HuggingFace 上为 `ef-rl*.parquet`，下载时自动重命名为 `e2c-rl*.parquet`

**换源机制：**
- `--mirror` - 使用镜像源 (hf-mirror.com，国内访问更快)
- `--no-mirror` - 强制使用官方源 (huggingface.co，默认)
- 环境变量 `HF_ENDPOINT` - 设置为包含 `hf-mirror.com` 的 URL 将自动使用镜像源
- 优先级：命令行参数 > 环境变量 > 默认值（官方源）

**评估数据集包括：**
- 数学类：aime24, aime25, amc23, gsm8k, math-algebra, math500, minerva, olympiad_bench
- 医学类：anatomy, clinical_knowledge, college_biology, college_medicine, medical_genetics, medmcqa, medqa, professional_medicine

所有评估数据下载到 `data/evaluation/` 目录（统一结构，不区分子目录）

#### `rename_rl_files.sh`
重命名已下载的 RL 文件（旧文件名转换）

```bash
# 如果之前下载过旧名称的文件（ef-rl*），使用此脚本重命名
bash rename_rl_files.sh
```

**说明：** 此脚本用于将旧的 `ef-rl*.parquet` 文件名改为 `e2c-rl*.parquet`。新下载的文件会自动使用新名称，无需手动重命名。

---

### 模型训练

#### `e2c_sft.sh`
E2C 监督微调（SFT）

```bash
bash e2c_sft.sh
```

**环境变量：**
```bash
export MODEL_PATH="Qwen/Qwen3-8B"                    # 基础模型
export TRAIN_DATA="data/processed/sft/e2c-sft-train.parquet"
export OUTPUT_DIR="models/checkpoints/sft"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_NUM=4
bash e2c_sft.sh
```

#### `e2c_rl.sh`
E2C 强化学习训练（两阶段）

```bash
bash e2c_rl.sh              # 运行两阶段训练
bash e2c_rl.sh --stage 1    # 仅运行阶段 1
bash e2c_rl.sh --stage 2    # 仅运行阶段 2
```

**环境变量：**
```bash
export MODEL_PATH="models/checkpoints/sft/final"    # SFT 检查点
export N_GPUS=8
bash e2c_rl.sh
```

#### `ef-sft.sh`
领域适应微调（Exploration-Focused SFT）

```bash
bash ef-sft.sh
```

**环境变量：**
```bash
export MODEL_PATH="models/checkpoints/rl/stage2-main/final"
export TRAIN_DATA="data/processed/ef_sft/medical-train.parquet"
bash ef-sft.sh
```

---

### 模型评估

#### `eval.sh`
评估训练好的 E2C 模型

```bash
# 基础评估（GSM8K）
bash eval.sh

# 评估特定数据集
bash eval.sh --dataset math --sample 4

# 评估所有数学基准
bash eval.sh --dataset all --sample 8

# 使用自定义模型
bash eval.sh --model models/checkpoints/rl/stage2-main/final

# 使用 HuggingFace 模型
bash eval.sh --model <your-org>/Explore-Execute-Chain --subfolder 4B-Final
```

**选项：**
- `--model PATH` - 模型路径或 HuggingFace ID
- `--subfolder NAME` - HuggingFace 子文件夹（如适用）
- `--dataset NAME` - 数据集名称（gsm8k, math, aime24, aime25, all, med）
- `--sample N` - 每个问题的采样次数
- `--gpus N` - GPU 数量
- `--temp T` - 采样温度
- `--save-path PATH` - 结果保存路径

**可用数据集：**
- **数学**: `gsm8k`, `math`, `aime24`, `aime25`, `amc23`, `all`
- **医学**: `medqa`, `medmcqa`, `med` (所有医学基准)

---

## 🔧 使用示例

### 完整训练流程

```bash
# 1. 准备数据
bash prepare_all_data.sh

# 2. SFT 训练
bash e2c_sft.sh

# 3. RL 训练
export MODEL_PATH="models/checkpoints/sft/final"
bash e2c_rl.sh

# 4. 评估
export MODEL_PATH="models/checkpoints/rl/stage2-main/final"
bash eval.sh --dataset all --sample 4
```

### 仅使用预训练模型进行评估

```bash
# 使用 HuggingFace 上的 4B 模型
bash eval.sh --model <your-org>/Explore-Execute-Chain --subfolder 4B-Final

# 使用 8B 模型
bash eval.sh --model <your-org>/Explore-Execute-Chain --subfolder 8B-Final --dataset math
```

### 领域适应

```bash
# 先完成 RL 训练，然后在新领域上微调
export MODEL_PATH="models/checkpoints/rl/stage2-main/final"
export TRAIN_DATA="data/processed/ef_sft/medical-train.parquet"
bash ef-sft.sh
```

---

## 💡 提示

1. **所有脚本都可以在 `scripts/` 目录下直接运行**
   ```bash
   cd scripts
   bash eval.sh
   ```

2. **查看脚本配置**
   - 每个脚本开头都有默认配置
   - 可以通过环境变量覆盖

3. **GPU 设置**
   - 使用 `CUDA_VISIBLE_DEVICES` 控制可见 GPU
   - 使用 `CUDA_NUM` 或 `N_GPUS` 设置 GPU 数量

4. **检查点位置**
   - SFT: `models/checkpoints/sft/`
   - RL: `models/checkpoints/rl/stage1-warmup/`, `models/checkpoints/rl/stage2-main/`
   - EF-SFT: `models/checkpoints/ef_sft/`

---

## 🐛 故障排除

### 找不到训练数据？
```bash
bash prepare_all_data.sh
```

### 找不到模型？
- 确保之前的训练步骤已完成
- 或者使用 HuggingFace 模型：`--model <your-org>/Explore-Execute-Chain`

### GPU 内存不足？
- 减少 GPU 数量
- 调整批大小环境变量
- 使用梯度累积

---

**返回主文档**: [../README.md](../README.md)

