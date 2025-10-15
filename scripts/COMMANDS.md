# E2C Scripts Command Reference

Quick reference for all E2C scripts commands.

## Table of Contents

- [Setup & Preparation](#setup--preparation)
  - [Adding Special Tokens](#adding-special-tokens)
  - [Data Preparation](#data-preparation)
- [Training Commands](#training-commands)
- [Generation Commands](#generation-commands)
- [Evaluation Commands](#evaluation-commands)
- [Combined Eval Commands](#combined-eval-commands)
- [Common Parameters](#common-parameters)

---

## Setup & Preparation

### Adding Special Tokens

⚠️ **Important:** If using a locally downloaded base model, you must add E2C special tokens before training.

**Required tokens:** `<EXPLORATION>`, `</EXPLORATION>`, `<EXECUTION>`, `</EXECUTION>`

```bash
# Add tokens to a local model
python scripts/add_special_tokens.py --model_path /path/to/model

# Specify custom output path
python scripts/add_special_tokens.py \
  --model_path /path/to/qwen3-8b \
  --output_path /path/to/qwen3-8b-e2c

# Dry run (preview without changes)
python scripts/add_special_tokens.py \
  --model_path /path/to/model \
  --dry_run
```

**Note:** This step is **automatically handled** when using HuggingFace model IDs (e.g., `Qwen/Qwen3-8B`).

### Configuring Special Token IDs

⚠️ **Critical for RL Training:** You must configure the token IDs for `</EXPLORATION>` and `<EXECUTION>` in your training scripts.

**Quick verification:**
```bash
# Check your model's token IDs
python3 << 'EOF'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your/model/path", trust_remote_code=True)
print(f"</EXPLORATION>: {tokenizer.convert_tokens_to_ids('</EXPLORATION>')}")
print(f"<EXECUTION>: {tokenizer.convert_tokens_to_ids('<EXECUTION>')}")
EOF
```

**Default token IDs (for Qwen2.5/3 with E2C tokens):**
- `</EXPLORATION>`: 151672
- `<EXECUTION>`: 151673

**Configure in RL training:**
```bash
# Option 1: Set environment variables
export SPECIAL_TOKEN_1=151672  # </EXPLORATION>
export SPECIAL_TOKEN_2=151673  # <EXECUTION>
bash scripts/e2c_rl.sh

# Option 2: Edit scripts/e2c_rl.sh directly
# Find and modify these lines:
SPECIAL_TOKEN_1="${SPECIAL_TOKEN_1:-151672}"
SPECIAL_TOKEN_2="${SPECIAL_TOKEN_2:-151673}"
```

**📖 For detailed token configuration guide, see:** [`docs/E2C_TOKEN_CONFIGURATION.md`](../docs/E2C_TOKEN_CONFIGURATION.md)

### Data Preparation

```bash
# Prepare all data (SFT + RL)
bash scripts/prepare_all_data.sh

# Skip download (use existing data)
bash scripts/prepare_all_data.sh --skip-download

# Prepare only SFT data
bash scripts/prepare_all_data.sh --skip-rl

# Prepare only RL data
bash scripts/prepare_all_data.sh --skip-sft

# Custom worker count
bash scripts/prepare_all_data.sh --num-workers 16
```

---

## Training Commands

### E2C-SFT Training

```bash
# Auto-detect all GPUs and train
bash scripts/e2c_sft.sh

# Use specific GPUs
export CUDA_VISIBLE_DEVICES="0,1"
bash scripts/e2c_sft.sh

# Full customization
export MODEL_PATH="Qwen/Qwen3-8B"
export TRAIN_DATA="data/processed/sft/e2c-sft-train.parquet"
export VAL_DATA="data/processed/sft/e2c-sft-val.parquet"
export OUTPUT_DIR="models/checkpoints/sft_custom"
export TOTAL_TRAINING_STEPS=1000
export EXPERIMENT_NAME="my_experiment"
bash scripts/e2c_sft.sh

# Use local model (with special tokens already added)
export MODEL_PATH="/path/to/qwen3-8b-e2c"
bash scripts/e2c_sft.sh
```

### E2C-RL Training

```bash
# Two-stage RL training (default)
bash scripts/e2c_rl.sh

# Customize model path
export MODEL_PATH="models/checkpoints/sft/final"
bash scripts/e2c_rl.sh

# Run specific stage only
bash scripts/e2c_rl.sh --stage 1  # Warm-up stage
bash scripts/e2c_rl.sh --stage 2  # Main stage
```

### EF-SFT (Domain Adaptation)

```bash
# Run EF-SFT
bash scripts/ef-sft.sh

# Customize for specific domain
export TRAIN_DATA="data/processed/ef_sft/medical-train.parquet"
export OUTPUT_DIR="models/checkpoints/ef_sft_medical"
bash scripts/ef-sft.sh
```

---

## Generation Commands

Generate responses with specified model and dataset:

```bash
# Generate with model on GSM8K dataset
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --sample 4
```

---

## Evaluation Commands  

Evaluate pre-generated responses:

```bash
# Evaluate generated results
bash scripts/eval_only.sh --generation-path ./generations --dataset gsm8k
```

---

## Combined Eval Commands

One-step generation + evaluation:

```bash
# Run generation and evaluation together
bash scripts/eval.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --sample 4
```

---

## Common Parameters

### Model Parameters

```bash
--model MODEL_PATH          # Full model path (e.g., yourpath/Explore-Execute-Chain/8B-Final, 
                            # /path/to/local/model, Qwen/Qwen3-8B)
--checkpoint CHECKPOINT     # Optional checkpoint path to load weights
```

### Dataset Parameters

```bash
--dataset DATASET           # Dataset name
                           # Single: gsm8k, math, aime24, aime25, amc23, math500, minerva, olympiad_bench
                           # Medical: clinical_knowledge, college_biology, college_medicine, 
                           #          medical_genetics, professional_medicine, anatomy, medqa, medmcqa
                           # Groups: all (all math), med (all medical)
```

### Generation Parameters

```bash
--sample N                  # Number of samples per question (default: 1)
--temp TEMPERATURE          # Sampling temperature (default: 1.0)
--gpus N_GPUS              # Number of GPUs to use (default: 1)
--save-path PATH           # Output save path
```

### Evaluation Parameters

```bash
--generation-path PATH     # Path to generated results (eval_only.sh only)
--distributed              # Enable distributed evaluation (eval_only.sh only)
--seed SEED                # Random seed (default: 0)
```

---

## Complete Workflow Examples

### Example 1: Standard Evaluation Flow

```bash
# Step 1: Generate on GSM8K with 4 samples
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --sample 4

# Step 2: Evaluate the generations
bash scripts/eval_only.sh --generation-path ./generations --dataset gsm8k

# Step 3: View results
cat evaluation/e2c-eval/gsm8k/static_0_merged.json
```

### Example 2: Multi-Dataset Benchmark

```bash
# Generate on all math benchmarks
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset all --sample 8 --gpus 4

# Evaluate all benchmarks
bash scripts/eval_only.sh --generation-path ./generations --dataset all

# View overall statistics
cat evaluation/e2c-eval/overall_static_0.json
```

### Example 3: Temperature Comparison

```bash
# Generate with different temperatures
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --temp 0.5 --sample 8 --save-path generations/temp0.5
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --temp 0.7 --sample 8 --save-path generations/temp0.7
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --temp 1.0 --sample 8 --save-path generations/temp1.0

# Evaluate each
bash scripts/eval_only.sh --generation-path generations/temp0.5 --dataset gsm8k --save-path evaluation/temp0.5
bash scripts/eval_only.sh --generation-path generations/temp0.7 --dataset gsm8k --save-path evaluation/temp0.7
bash scripts/eval_only.sh --generation-path generations/temp1.0 --dataset gsm8k --save-path evaluation/temp1.0

# Compare results
echo "=== Temperature 0.5 ===" && cat evaluation/temp0.5/gsm8k/static_0_merged.json
echo "=== Temperature 0.7 ===" && cat evaluation/temp0.7/gsm8k/static_0_merged.json
echo "=== Temperature 1.0 ===" && cat evaluation/temp1.0/gsm8k/static_0_merged.json
```

### Example 4: Model Comparison

```bash
# Generate with 4B model
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/4B-Final --dataset gsm8k --sample 8 --save-path generations/4B

# Generate with 8B model
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --sample 8 --save-path generations/8B

# Evaluate both
bash scripts/eval_only.sh --generation-path generations/4B --dataset gsm8k --save-path evaluation/4B
bash scripts/eval_only.sh --generation-path generations/8B --dataset gsm8k --save-path evaluation/8B

# Compare
diff <(cat evaluation/4B/gsm8k/static_0_merged.json) <(cat evaluation/8B/gsm8k/static_0_merged.json)
```

### Example 5: Quick One-Step Evaluation

```bash
# Use original script for quick testing
bash scripts/eval.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --sample 4

# View results immediately
cat evaluation/e2c-eval/gsm8k/static_0_merged.json
```

---

## Tips and Best Practices

### 1. Choosing Between Separate and Combined Scripts

**Use `generate.sh` + `eval_only.sh` when:**
- Running large-scale experiments
- Need to evaluate multiple times with different criteria
- Comparing different models/settings
- Want to save computation resources

**Use `eval.sh` when:**
- Quick testing
- One-off evaluations
- Don't need to re-evaluate

### 2. Resource Management

```bash
# For large models, use multiple GPUs
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset all --gpus 4

# For limited GPU memory, reduce batch size via environment variable
BATCH_SIZE=4 bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset math
```

### 3. Organizing Results

```bash
# Use descriptive save paths
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --save-path generations/8B-gsm8k-s8-t07 --sample 8 --temp 0.7

# Keep generation and evaluation paths organized
bash scripts/eval_only.sh --generation-path generations/8B-gsm8k-s8-t07 --save-path evaluation/8B-gsm8k-s8-t07
```

### 4. Parallel Runs

```bash
# Run multiple datasets in parallel (different terminals)
# Terminal 1
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k --save-path generations/gsm8k

# Terminal 2
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset math --save-path generations/math

# Terminal 3
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset aime24 --save-path generations/aime24
```

---

## Environment Variables

You can also use environment variables to set parameters:

```bash
# Set model path
export MODEL_PATH=yourpath/Explore-Execute-Chain/8B-Final

# Set generation parameters
export SAMPLE_NUM=8
export TEMPERATURE=0.7
export N_GPUS=2

# Run with environment variables
bash scripts/generate.sh --dataset gsm8k
```

---

## View Results

### View Statistics

```bash
# View evaluation statistics (formatted)
cat evaluation/e2c-eval/gsm8k/static_0_merged.json | python -m json.tool

# View specific field
cat evaluation/e2c-eval/gsm8k/static_0_merged.json | jq '.avg_success'

# View overall statistics for all datasets
cat evaluation/e2c-eval/overall_static_0.json | python -m json.tool
```

### View Detailed Results

```bash
# View detailed results
less evaluation/e2c-eval/gsm8k/result_0_merged.json

# Count successful samples
cat evaluation/e2c-eval/gsm8k/result_0_merged.json | jq '[.[] | select(.best_success == 1)] | length'

# View failed questions
cat evaluation/e2c-eval/gsm8k/result_0_merged.json | jq '[.[] | select(.best_success == 0) | .question]'
```

---

## Troubleshooting

### Common Issues

```bash
# If generation fails due to CUDA out of memory
# Reduce batch size
BATCH_SIZE=2 bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset math

# If eval_only.sh can't find generations
# Check the generation path
ls -la ./generations/gsm8k/

# If you need to resume interrupted generation
# The script automatically resumes from checkpoint (if resume=true in config)
bash scripts/generate.sh --model yourpath/Explore-Execute-Chain/8B-Final --dataset gsm8k  # Will auto-resume
```

---

## Additional Resources

- Full documentation: `docs/GENERATE_EVAL_USAGE.md`
- Configuration files: `e2c/config/`
- Source code: `e2c/inference/`

