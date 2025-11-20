# Explore–Execute Chain (E2C)

This repository provides the implementation of our paper:
**Explore–Execute Chain: Towards an Efficient Structured Reasoning Paradigm**

---

## 📖 Overview

Large Language Models (LLMs) often intertwine high-level planning with low-level execution, limiting both efficiency and interpretability.
The **Explore–Execute Chain (E2C)** framework addresses this challenge by **decoupling reasoning into two distinct stages**:

1. **Exploration** — Generate lightweight reasoning sketches (plans).
2. **Execution** — Select one or a few promising plans and execute them faithfully.

**Benefits:**

* **Efficiency** — Short explorations reduce computation while preserving reasoning coverage.
* **Interpretability** — Exploration traces are explicit and auditable.
* **Adaptability** — Exploration can be domain-adapted with minimal supervision.

---

## 🚀 Key Features

* **Two-stage reasoning**: Exploration → Execution
* **Complete training pipeline**:

  * **E2C-SFT** — Supervised fine-tuning on synthetic exploration–execution pairs ✅ **Ready**
  * **E2C-RL** — Reinforcement learning with token-level weighting ✅ **Ready**
* **Efficient domain adaptation (EF-SFT)** — Train with exploration-only data ✅ **Ready**
* **Test-time scaling (TTS)** — Sample multiple explorations and aggregate via clustering / LM fusion 🔄 **Under continuous integration**
* **Benchmarks**:
  * **Mathematics**: AIME’24/25, MATH500, Olympiad, Minerva
  * **Medical reasoning**: MedQA, MedMCQA, MMLU medical subsets
---

## 📂 Repository Structure

```
.
├── scripts/                    # All executable scripts
│   ├── prepare_all_data.sh     # Data preparation
│   ├── download_datasets.sh    # Download datasets
│   ├── download_model.sh       # Model download
│   ├── eval.sh                 # Model evaluation
│   ├── e2c_sft.sh             # SFT training
│   ├── e2c_rl.sh              # RL training (PPO)
│   ├── e2c_dapo.sh            # DAPO training
│   └── ef-sft.sh              # Domain adaptation
├── e2c/                       # Core E2C framework
│   ├── inference/             # Inference and evaluation
│   ├── util/                  # Utility functions
│   └── config/                # Configuration files
├── verl/                      # Training framework (VERL-based)
├── data/                      # Dataset preparation and evaluation data
├── models/                    # Model checkpoints and pretrained models
│   ├── pretrained/            # Pretrained models
│   └── released/              # Released model versions
├── generations/               # Generated outputs from models
├── evaluation/                # Evaluation results and metrics
├── outputs/                   # Training and experiment outputs
├── example_inference.py       # Example inference script
├── example_interactive.py     # Interactive example script
└── README.md
```

---

## ⚙️ Setup

### Requirements

See `e2c/verl/README.md` for details.

Install dependencies via:

```bash
pip install -r requirements.txt
```

> All dependencies are included for both training and inference. See `verl/README.md` for detailed setup instructions.

---

## 🚀 Quick Start - Inference Example

Use E2C models directly from Hugging Face (no manual download required):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model directly from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    "anomyous-author/Explore-Execute-Chain",
    subfolder="4B-Final",  # or "8B-Final" for the 8B model
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "anomyous-author/Explore-Execute-Chain",
    subfolder="4B-Final"
)

# Example math problem
problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. 
She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
How much in dollars does she make every day at the farmers' market?"""

# Create prompt and generate
messages = [{"role": "user", "content": problem}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=" * 50)
print("E2C Reasoning:")
print("=" * 50)
print(response)
```

**Model Options:**
- **4B Model**: `subfolder="4B-Final"` (~8GB)
- **8B Model**: `subfolder="8B-Final"` (~16GB)

---

## 📦 Data Preparation

### Quick Start

Prepare training data in one command:

```bash
# Basic usage
bash scripts/prepare_all_data.sh

# Use mirror source (recommended for China)
bash scripts/prepare_all_data.sh --mirror

# Skip download, process existing data only
bash scripts/prepare_all_data.sh --skip-download

# Prepare only SFT data
bash scripts/prepare_all_data.sh --skip-rl
```

This will:
- ✅ Download datasets from [HuggingFace](https://huggingface.co/datasets/anomyous-author/Explore-Execute-Chain-Datasets)
- ✅ Process SFT training data
- ✅ Process RL training data

**Time:** ~30-60 minutes | **Storage:** ~10 GB

### Download Datasets Only

If you only need to download raw datasets without processing:

```bash
# Download all datasets (SFT + RL + Evaluation)
bash scripts/download_datasets.sh

# Download specific dataset types
bash scripts/download_datasets.sh --dataset sft    # Only SFT data
bash scripts/download_datasets.sh --dataset rl      # Only RL data
bash scripts/download_datasets.sh --dataset eval   # Only evaluation datasets (16 datasets)
```

**🌐 Mirror Support (China):**

```bash
# Use --mirror flag
bash scripts/download_datasets.sh --mirror
bash scripts/prepare_all_data.sh --mirror

# Or set environment variable
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_datasets.sh
```

**Dataset Information:**
- **SFT Data**: `e2c-sft.parquet` (77.7 MB) → `data/raw/sft/`
- **RL Data**: 
  - `e2c-rl.parquet` (19.4 MB) → `data/raw/rl/`
  - `e2c-rl-valid.parquet` (706 KB) → `data/raw/rl/`
- **Evaluation Data**: 16 datasets → `data/evaluation/`
  - **Math**: aime24, aime25, amc23, gsm8k, math-algebra, math500, minerva, olympiad_bench
  - **Medical**: anatomy, clinical_knowledge, college_biology, college_medicine, medical_genetics, medmcqa, medqa, professional_medicine

**Note:** The datasets are hosted at `anomyous-author/Explore-Execute-Chain-Datasets` on HuggingFace. The download script automatically handles file naming (e.g., `ef-rl.parquet` → `e2c-rl.parquet`).

---

## 🤖 Models

### E2C Released Models

E2C models are available on HuggingFace at `anomyous-author/Explore-Execute-Chain`:

| Model | Parameters | Size | Subfolder | Link |
|-------|-----------|------|-----------|------|
| E2C-Qwen3-4B | 4B | ~8 GB | `4B-Final` | [HuggingFace](https://huggingface.co/anomyous-author/Explore-Execute-Chain) |
| E2C-Qwen3-8B | 8B | ~16 GB | `8B-Final` | [HuggingFace](https://huggingface.co/anomyous-author/Explore-Execute-Chain) |

### Download Models

Models are automatically downloaded when you use them in code. However, you can also pre-download them:

**Using Script (Recommended):**

```bash
# Download 4B model using mirror (recommended for China)
bash scripts/download_model.sh --subfolder 4B-Final --mirror

# Download 8B model using mirror
bash scripts/download_model.sh --subfolder 8B-Final --mirror

# Download from official source
bash scripts/download_model.sh --subfolder 4B-Final
```

**🌐 Mirror Support (China):**

```bash
# Use script with --mirror flag
bash scripts/download_model.sh --subfolder 4B-Final --mirror

# Or set environment variable
export HF_ENDPOINT=https://hf-mirror.com
# Then use transformers normally
```

### Base Models for Training

If you want to train your own E2C models, use these base models:

| Model | Parameters | Link |
|-------|-----------|------|
| Qwen3-4B | 4B | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen3-8B | 8.2B | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) |
| Llama-3.1-8B-Instruct | 8B | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |

> ⚠️ **Important for Local Models:** If you use a locally downloaded base model, you **must** manually add the following special tokens to the tokenizer and resize the model embedding table before training:
> 
> **Required Special Tokens:**
> - `<EXPLORATION>` - Marks the start of exploration phase
> - `</EXPLORATION>` - Marks the end of exploration phase
> - `<EXECUTION>` - Marks the start of execution phase
> - `</EXECUTION>` - Marks the end of execution phase
>
> **How to add tokens:**
> ```python
> from transformers import AutoTokenizer, AutoModelForCausalLM
> 
> # Load model and tokenizer
> model = AutoModelForCausalLM.from_pretrained("path/to/your/model")
> tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")
> 
> # Add special tokens
> special_tokens = ["<EXPLORATION>", "</EXPLORATION>", "<EXECUTION>", "</EXECUTION>"]
> num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
> 
> # Resize model embedding table
> model.resize_token_embeddings(len(tokenizer))
> 
> # Save updated model and tokenizer
> model.save_pretrained("path/to/updated/model")
> tokenizer.save_pretrained("path/to/updated/model")
> ```
>
> **Note:** This step is **automatically handled** when using HuggingFace model IDs (e.g., `Qwen/Qwen3-8B`) in the training scripts. Only needed for pre-downloaded local models.


---

## 🏋️ Training Pipeline

E2C training follows a two-stage pipeline: **SFT → RL**. Optional EF-SFT for domain adaptation.

### Prerequisites

**Prepare Data**
```bash
bash scripts/prepare_all_data.sh
```

**Note:** Base models (Qwen3-8B, etc.) will be automatically downloaded by the training scripts from Hugging Face.

### Stage 1: Supervised Fine-Tuning (E2C-SFT)

Train the model to generate exploration-execution chains:

```bash
# Run SFT training with default settings
bash scripts/e2c_sft.sh

# Or customize model/data
export MODEL_PATH="Qwen/Qwen3-8B"  # HuggingFace model ID
export TRAIN_DATA="data/processed/sft/e2c-sft-train.parquet"
bash scripts/e2c_sft.sh
```

**Training Configuration:**
- Base model: Qwen3-8B (auto-downloaded from HF)
- Training data: ~50K exploration-execution pairs
- Training time: ~8 hours on 4×A100
- Output: SFT checkpoint in `models/checkpoints/sft/`

### Stage 2: Reinforcement Learning

**Option A: GRPO Training (e2c_rl.sh)**

```bash
# Two-stage RL training
export MODEL_PATH="models/checkpoints/sft/final"
export N_GPUS=8
bash scripts/e2c_rl.sh

# Run specific stage
bash scripts/e2c_rl.sh --stage 2

# Configure special tokens (if needed)
export SPECIAL_TOKEN_1=151672  # </EXPLORATION>
export SPECIAL_TOKEN_2=151673  # <EXECUTION>
bash scripts/e2c_rl.sh
```

**Option B: DAPO Training (e2c_dapo.sh)**

```bash
# DAPO with token operations
export MODEL_PATH="models/checkpoints/sft/final"
export N_GPUS=8
bash scripts/e2c_dapo.sh

# Run specific stage
bash scripts/e2c_dapo.sh --stage 2
```

**Training Configuration:**
- **Stage 1**: 32 rollouts, 1 epoch
- **Stage 2**: 8 rollouts, 2 epochs
- Training data: 17K problems
- Output: `models/checkpoints/rl/stage2-main/` or `models/checkpoints/dapo/stage2-main/`

### Optional: Efficient Domain Adaptation (EF-SFT)

Adapt to new domains with exploration-only data (requires only ~3.5% tokens):

```bash
# Run EF-SFT (uses RL checkpoint by default)
bash scripts/ef-sft.sh

# Or customize for your domain
export TRAIN_DATA="data/processed/ef_sft/medical-train.parquet"
bash scripts/ef-sft.sh
```

**Use Cases:**
- Domain adaptation (medical, code, etc.)
- Quick tuning with limited data
- Exploration strategy refinement

---

## 🔍 Evaluation

### Option 1: Using Scripts (Recommended)

Use the convenient shell scripts for quick evaluation:

```bash
# Quick test on GSM8K with 8B-Final
bash scripts/eval.sh --subfolder 8B-Final

# Evaluate on specific dataset
bash scripts/eval.sh --subfolder 8B-Final --dataset math --sample 4

# Evaluate all math benchmarks
bash scripts/eval.sh --subfolder 8B-Final --dataset all --sample 8

# Use custom model
bash scripts/eval.sh --model /path/to/model --dataset gsm8k
```

### Option 2: Direct Python Scripts (Advanced)

For more control, use Python scripts directly with Hydra configuration:

#### Generation

Generate model responses using the `generate.py` script:

```bash
# Basic usage (uses default config from e2c/config/generate.yaml)
python e2c/inference/generate.py \
    --config-path=e2c/config \
    --config-name=generate

# Customize via command line overrides
python e2c/inference/generate.py \
    --config-path=e2c/config \
    --config-name=generate \
    generation.dataset=['gsm8k'] \
    generation.sample_num=8 \
    generation.temperature=0.7 \
    model.model_path="anomyous-author/Explore-Execute-Chain" \
    model.checkpoint_path="4B-Final"

# Use VLLM backend for faster inference
python e2c/inference/generate.py \
    --config-path=e2c/config \
    --config-name=generate \
    model.type=vllm \
    model.model_path="anomyous-author/Explore-Execute-Chain" \
    model.checkpoint_path="8B-Final" \
    generation.dataset=['math'] \
    generation.sample_num=4
```

**Generation Configuration Options:**
- `generation.dataset`: Dataset name(s) - `['gsm8k']`, `['math']`, `['all']`, `['med']`, etc.
- `generation.sample_num`: Number of samples per question (default: 5)
- `generation.temperature`: Sampling temperature (default: 1.0)
- `generation.save_path`: Output directory (default: `./generations`)
- `model.model_path`: Model path or HuggingFace ID (e.g., `anomyous-author/Explore-Execute-Chain`)
- `model.checkpoint_path`: Subfolder for HuggingFace models (e.g., `4B-Final`, `8B-Final`)
- `model.type`: Backend type - `hf` (HuggingFace) or `vllm` (VLLM, faster)

#### Evaluation

Evaluate generated responses using the `eval.py` script:

```bash
# Basic usage (evaluates generations from default path)
python e2c/inference/eval.py \
    --config-path=e2c/config \
    --config-name=eval \
    eval.generation_path=./generations \
    eval.dataset=['gsm8k'] \
    eval.save_path=./evaluation

# Evaluate multiple datasets
python e2c/inference/eval.py \
    --config-path=e2c/config \
    --config-name=eval \
    eval.generation_path=./generations \
    eval.dataset=['gsm8k','math','aime24'] \
    eval.save_path=./evaluation

# Evaluate all math benchmarks
python e2c/inference/eval.py \
    --config-path=e2c/config \
    --config-name=eval \
    eval.generation_path=./generations \
    eval.dataset=['all'] \
    eval.save_path=./evaluation
```

**Evaluation Configuration Options:**
- `eval.generation_path`: Path to generated results directory (required)
- `eval.dataset`: Dataset name(s) to evaluate
- `eval.save_path`: Output directory for evaluation results
- `eval.seed`: Random seed (should match generation seed)

**Note:** The `eval.py` script expects a `config/eval.yaml` file. If it doesn't exist, you can create one based on `eval_only.yaml` or use command-line overrides.

#### Two-Step Workflow Example

```bash
# Step 1: Generate responses
python e2c/inference/generate.py \
    --config-path=e2c/config \
    --config-name=generate \
    generation.dataset=['gsm8k'] \
    generation.sample_num=8 \
    model.model_path="anomyous-author/Explore-Execute-Chain" \
    model.checkpoint_path="8B-Final" \
    generation.save_path=./generations/gsm8k-s8

# Step 2: Evaluate generated results
python e2c/inference/eval.py \
    --config-path=e2c/config \
    --config-name=eval \
    eval.generation_path=./generations/gsm8k-s8 \
    eval.dataset=['gsm8k'] \
    eval.save_path=./evaluation/gsm8k-s8
```

**Available Datasets:**
- **Math**: `gsm8k`, `math`, `aime24`, `aime25`, `amc23`, `math500`, `minerva`, `olympiad_bench`, `all`
- **Medical**: `medqa`, `medmcqa`, `clinical_knowledge`, `college_biology`, `college_medicine`, `medical_genetics`, `professional_medicine`, `anatomy`, `med` (all medical)

---

## 📊 Results

E2C consistently improves over GRPO baselines while reducing computation.

* **Mathematics**: On AIME'24, Qwen3-4B + E2C-(SFT+RL) shows **+8.7% accuracy gain** over GRPO
* **Medical reasoning**: EF-SFT adapts with only **3.5% of training tokens**, outperforming standard SFT
* **Efficiency**: E2C reduces computation while maintaining reasoning quality through structured exploration

> Results are reproducible with the provided training pipeline and released models.

---

## 🙏 Credits

This work builds upon the **VERL** framework.
We thank the authors for releasing such a flexible and powerful platform.

---

## 🧾 License

This project is licensed under the **MIT License**.
See the [LICENSE](verl/LICENSE) file for details.

---
