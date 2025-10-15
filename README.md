# Explore–Execute Chain (E2C)

This repository provides the implementation of our paper:
**Explore–Execute Chain: Towards an Efficient Structured Reasoning Paradigm**
*Kaisen Yang, Lixuan He, Rushi Shah, Kaicheng Yang, Qinwei Ma, Dianbo Liu, Alex Lamb*


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

⚠️ **Repository Status:** This repository is now **fully released** with comprehensive training and inference capabilities.

* ✅ **Models released**: E2C-Qwen3-4B and E2C-Qwen3-8B models are available on HuggingFace
* ✅ **Training pipeline**: Complete SFT, RL, and EF-SFT training scripts are ready to use
* ✅ **Inference & evaluation**: Full evaluation pipeline with multiple benchmarks
* 🔄 **TTS (Test-Time Scaling)**: Advanced TTS features are under continuous integration and improvement

---

## 📝 Current Progress

**Released / Fully Usable:**

* ✅ **Pretrained E2C models**: Qwen3-4B and Qwen3-8B models on HuggingFace
* ✅ **Complete training pipeline**: SFT, RL, and EF-SFT training scripts
* ✅ **Evaluation datasets**: Mathematics, Medical, and other benchmarks
* ✅ **Inference & evaluation**: Full pipeline for model testing and benchmarking
* ✅ **Data preparation**: Automated data processing and dataset preparation
* ✅ **Special token handling**: Automatic token configuration for training

**Under Continuous Integration:**

* 🔄 **Advanced TTS features**: Enhanced test-time scaling with clustering and fusion
* 🔄 **Extended exploration strategies**: Additional reasoning mechanisms
* 🔄 **Performance optimizations**: Further efficiency improvements

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
├── scripts/                    # 🔧 All executable scripts (run from here!)
│   ├── prepare_all_data.sh     # Data preparation
│   ├── download_datasets.sh    # Download datasets
│   ├── eval.sh                 # Model evaluation
│   ├── e2c_sft.sh             # SFT training
│   ├── e2c_rl.sh              # RL training
│   ├── ef-sft.sh              # Domain adaptation
│   ├── generate.sh            # Generation scripts
│   └── COMMANDS.md            # Command reference
├── e2c/                       # Core E2C framework
│   ├── inference/             # Test-time scaling and evaluation ✅ ready
│   ├── util/                  # Utility functions and model helpers ✅ ready
│   └── config/                # Configuration files ✅ ready
├── verl/                      # Training framework (VERL-based)
│   ├── examples/              # Training examples and recipes ✅ ready
│   ├── verl/                  # Core training components ✅ ready
│   └── tests/                 # Test suites ✅ ready
├── data/                      # Dataset preparation and evaluation data
├── models/                    # Model checkpoints and pretrained models
│   ├── checkpoints/           # Training checkpoints
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
    "KaisenYang/Explore-Execute-Chain",
    subfolder="4B-Final",  # or "8B-Final" for the 8B model
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "KaisenYang/Explore-Execute-Chain",
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
- **4B Model**: `subfolder="4B-Final"` (~8GB, faster inference)
- **8B Model**: `subfolder="8B-Final"` (~16GB, better performance)

**More Examples:**
- 🔧 **Complete training pipeline**: See [Training Pipeline](#️-training-pipeline) section below
- 📊 **Evaluation scripts**: `e2c/inference/` and `scripts/` directory
- 📖 **Documentation**: See respective README files in each directory
- 🚀 **Quick training**: Run `bash scripts/e2c_sft.sh` to start training

---

## 📦 Data Preparation

Prepare training data in one command:

```bash
bash scripts/prepare_all_data.sh
```

This will:
- ✅ Download datasets from [HuggingFace](https://huggingface.co/datasets/KaisenYang/Explore-Execute-Chain-Datasets)
- ✅ Process SFT training data
- ✅ Process RL training data

**Time:** ~30-60 minutes | **Storage:** ~10 GB

**Quick download only:**
```bash
bash scripts/download_datasets.sh

# For users in China (faster with mirror):
bash scripts/download_datasets.sh --mirror
```

**🌐 Mirror Support:**
- Use `--mirror` flag to download from hf-mirror.com (faster for users in China)
- Set environment variable: `export HF_ENDPOINT=https://hf-mirror.com`

---

## 🤖 Models

### E2C Released Models

Use E2C models directly from Hugging Face (no download needed):

| Model | Parameters | Size | Subfolder | Link |
|-------|-----------|------|-----------|------|
| E2C-Qwen3-4B | 4B | ~8 GB | `4B-Final` | [HuggingFace](https://huggingface.co/KaisenYang/Explore-Execute-Chain) |
| E2C-Qwen3-8B | 8B | ~16 GB | `8B-Final` | [HuggingFace](https://huggingface.co/KaisenYang/Explore-Execute-Chain) |

See [Quick Start](#-quick-start---inference-example) above for usage examples.

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

**For more details:** See the [Training Pipeline](#️-training-pipeline) section above for complete training instructions.

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

### Stage 2: Reinforcement Learning (E2C-RL)

Fine-tune with RL to improve reasoning quality (two-stage training):

```bash
# Run two-stage RL training (default)
bash scripts/e2c_rl.sh

# Or customize with environment variables
export MODEL_PATH="models/checkpoints/sft/final"
export TRAIN_DATA="data/processed/rl/e2c-rl-train.parquet"
export VAL_DATA="data/processed/rl/e2c-rl-val.parquet"
bash scripts/e2c_rl.sh

# Or run specific stage only
bash scripts/e2c_rl.sh --stage 2  # Only run stage 2
```

**⚠️ Important: Configure Special Token IDs**

Before RL training, verify and configure the token IDs for `</EXPLORATION>` and `<EXECUTION>`:

```bash
# Check your model's token IDs
python3 << 'EOF'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your/model/path", trust_remote_code=True)
print(f"</EXPLORATION>: {tokenizer.convert_tokens_to_ids('</EXPLORATION>')}")
print(f"<EXECUTION>: {tokenizer.convert_tokens_to_ids('<EXECUTION>')}")
EOF

# Configure for training (default: 151672, 151673 for Qwen2.5/3)
export SPECIAL_TOKEN_1=151672  # </EXPLORATION>
export SPECIAL_TOKEN_2=151673  # <EXECUTION>
bash scripts/e2c_rl.sh
```

**Training Configuration:**
- **Stage 1 (Warm-up)**: 32 rollouts, 1 epoch, no constraints
- **Stage 2 (Main)**: 8 rollouts, 2 epochs, with constraints
- Starting point: E2C-SFT checkpoint
- Training data: ~20K problems with ground truth
- Training time: ~12-16 hours on 8×A100
- Output: Final E2C model in `models/checkpoints/rl/stage2-main/`

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

### Training Tips

**GPU Memory Requirements:**
- **Qwen3-4B**: 4×RTX 3090 (24GB each) or 2×A100
- **Qwen3-8B**: 4×A100 (40GB each) recommended
- **Llama-3.1-8B**: Similar to Qwen3-8B

**Hyperparameter Tuning:**
- Learning rate: 1e-6 (SFT), 5e-7 (RL)
- Batch size: Adjust based on GPU memory
- Gradient accumulation: Use to simulate larger batches

**Checkpointing:**
- SFT: Saves every 500 steps
- RL: Saves every epoch
- Best checkpoint selected by validation accuracy

---

## 🔍 Evaluation

### Option 1: One-Step Evaluation (Quick)

Generate and evaluate in one command:

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

### Option 2: Two-Step Evaluation (Flexible)

For multiple evaluations or analysis, separate generation and evaluation:

```bash
# Step 1: Generate responses once
bash scripts/generate.sh --subfolder 8B-Final --dataset gsm8k --sample 8

# Step 2: Evaluate (can run multiple times with different criteria)
bash scripts/eval_only.sh \
  --generation-path ./generations \
  --dataset gsm8k
```

**Available Datasets:**
- **Math**: `gsm8k`, `math`, `aime24`, `aime25`, `amc23`, `math500`, `minerva`, `olympiad_bench`, `all`
- **Medical**: `medqa`, `medmcqa`, `clinical_knowledge`, `college_biology`, `med` (all medical)

**💡 For complete command reference, see:** [`scripts/COMMANDS.md`](scripts/COMMANDS.md)

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

## 📌 Notes

* Repository is **fully released** with complete training and inference capabilities.
* **Training scripts** are ready to use for SFT, RL, and EF-SFT training.
* **TTS features** are under continuous integration with advanced clustering and fusion capabilities.
* Hyperparameters and prompt templates are provided in the paper appendices.
* Users can **directly run model inference** and **full training pipeline** with released datasets and code.

---

## 📜 Citation

**BibTeX**

@misc{yang2025exploreexecutechainefficientstructured,
      title={Explore-Execute Chain: Towards an Efficient Structured Reasoning Paradigm}, 
      author={Kaisen Yang and Lixuan He and Rushi Shah and Kaicheng Yang and Qinwei Ma and Dianbo Liu and Alex Lamb},
      year={2025},
      eprint={2509.23946},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.23946}, 
}

**APA**

> Yang, K., He, L., Shah, R., Yang, K., Ma, Q., Liu, D., & Lamb, A. (2026). *Explore–Execute Chain: Towards an Efficient Structured Reasoning Paradigm*. (Under review)

**MLA**

> Yang, Kaisen, et al. "Explore–Execute Chain: Towards an Efficient Structured Reasoning Paradigm." Under review.

---

## 🧾 License

This project is licensed under the **MIT License**.
See the [LICENSE](verl/LICENSE) file for details.

---
