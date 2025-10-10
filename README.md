# Explore–Execute Chain (E2C)

This repository provides the implementation of our paper:
**Explore–Execute Chain: Towards an Efficient Structured Reasoning Paradigm**
*Kaisen Yang, Lixuan He, Rushi Shah, Kaicheng Yang, Qinwei Ma, Dianbo Liu, Alex Lamb*

> *Under review at ICLR 2026*

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

⚠️ **Repository Status:** This repository is under active development.

* ✅ Model inference and evaluation with released datasets are **fully usable**.
* 🔄 Training scripts (SFT, RL, EF-SFT) and test-time scaling (TTS) are **still being organized**.

---

## 📝 Current Progress

**Released / Usable:**

* ✅ Pretrained E2C models (Qwen3-4B / Qwen3-8B)
* ✅ Evaluation datasets (Mathematics, Medical, and other benchmarks)
* ✅ Core inference scripts for running reasoning on released models

**In Development:**

* 🔄 Training scripts for SFT, RL, and EF-SFT
* 🔄 Test-time scaling (TTS) features
* 🔄 Additional exploration strategies and execution mechanisms
* 🔄 Extended experiments and reproducibility utilities

---

## 🚀 Key Features

* **Two-stage reasoning**: Exploration → Execution
* **Training pipeline**:

  * **E2C-SFT** — Supervised fine-tuning on synthetic exploration–execution pairs *(under development)*
  * **E2C-RL** — Reinforcement learning with token-level weighting *(under development)*
* **Efficient domain adaptation (EF-SFT)** — Train with exploration-only data *(under development)*
* **Test-time scaling (TTS)** — Sample multiple explorations and aggregate via clustering / LM fusion *(under development)*
* **Benchmarks**:

  * **Mathematics**: AIME’24/25, MATH500, Olympiad, Minerva
  * **Medical reasoning**: MedQA, MedMCQA, MMLU medical subsets

---

## 📂 Repository Structure

```
.
├── data/           # Dataset preparation scripts and evaluation data (usable)
├── e2c/            # Core E2C framework
│   ├── sft/        # Supervised fine-tuning (E2C-SFT, EF-SFT) ⚠️ under development
│   ├── rl/         # Reinforcement learning (E2C-RL) ⚠️ under development
│   └── inference/  # Test-time scaling and evaluation ✅ usable
├── prompts/        # Prompt templates
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

> Only inference-related dependencies are required to run released models.

---

## 📥 Model & Dataset

* **Pretrained Models**:
  👉 [KaisenYang/Explore-Execute-Chain](https://huggingface.co/KaisenYang/Explore-Execute-Chain)

* **Datasets**:
  👉 [KaisenYang/Explore-Execute-Chain-Datasets](https://huggingface.co/datasets/KaisenYang/Explore-Execute-Chain-Datasets)

After downloading, place the evaluation data in the `data/` folder.

---

## 🏋️ Training ⚠️ Under Development

### Stage 1: Supervised Fine-Tuning (E2C-SFT)

```bash
bash e2c/sft/train_sft.sh  # ⚠️ currently being organized
```

### Stage 2: Reinforcement Learning (E2C-RL)

```bash
bash e2c/rl/train_rl.sh  # ⚠️ currently being organized
```

### Efficient Adaptation (EF-SFT)

```bash
bash e2c/sft/train_ef_sft.sh  # ⚠️ currently being organized
```

---

## 🔍 Evaluation ✅ Usable

Run evaluation and test-time inference:

```bash
bash e2c/inference/evaluate.sh
```

Supported strategies (usable):

* **E2C-Select (Self LM-Judge)**
* **E2C-Select (Semantic Cluster)**
* **E2C-SC (Self-Consistency)**
* **E2C-RP (Random Plan)**

---

## 📊 Results

E2C consistently improves over GRPO baselines while reducing computation.

* **Mathematics**: On AIME’24, Qwen3-4B + E2C-(SFT+RL) shows **+8.7% accuracy gain** over GRPO
* **Medical reasoning**: EF-SFT adapts with only **3.5% of training tokens**, outperforming standard SFT

> Note: Results are from released models; full reproduction with training code is pending.

---

## 🙏 Credits

This work builds upon the **VERL** framework.
We thank the authors for releasing such a flexible and powerful platform.

---

## 📌 Notes

* Repository is **actively updated**; keep an eye on releases for training scripts and TTS.
* Hyperparameters and prompt templates are provided in the paper appendices.
* Users can **directly run model inference** with released datasets now.

---

## 📜 Citation

**BibTeX**

```bibtex
@inproceedings{yang2026explore,
  title={Explore-Execute Chain: Towards an Efficient Structured Reasoning Paradigm},
  author={Yang, Kaisen and He, Lixuan and Shah, Rushi and Yang, Kaicheng and Ma, Qinwei and Liu, Dianbo and Lamb, Alex},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  note={under review}
}
```

**APA**

> Yang, K., He, L., Shah, R., Yang, K., Ma, Q., Liu, D., & Lamb, A. (2026). *Explore–Execute Chain: Towards an Efficient Structured Reasoning Paradigm*. International Conference on Learning Representations (ICLR). (Under review)

**MLA**

> Yang, Kaisen, et al. "Explore–Execute Chain: Towards an Efficient Structured Reasoning Paradigm." *International Conference on Learning Representations (ICLR)*, 2026. Under review.

---

## 🧾 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---
