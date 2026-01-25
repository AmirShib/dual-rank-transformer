# Universal Dual-Task Ranking Transformer

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

An implementation of a multi-head Transformer for simultaneous **Occupation** and **Sector** ranking. This model leverages a composite loss function (Listwise Cross-Entropy + Pairwise Soft Margin) to optimize ranking performance across hierarchical classification tasks.

---

### Context & Usage Disclaimer

**This repository serves as a reference implementation and portfolio demonstration.**

The core architecture and training methodologies presented here were designed for a production system currently deployed in a **secure, air-gapped environment**. 

* **Code Adaptation:** This code has been sanitized and decoupled from internal infrastructure to demonstrate some of the architectural patterns, training loops, and loss functions used in the project.
* **Data Privacy:** No proprietary data, internal configurations, or sensitive logic from the production environment is included.
* **Optimization:** The production model utilizes domain-specific optimizations that are not reflected here for security reasons.

---

## Architecture

The model uses a shared Encoder-Only Transformer backbone (e.g., `alephbert-base` or `bert-base`) to encode input text, followed by two task-specific heads.

### Key Components:
1.  **Shared Backbone:** Extracts contextual embeddings from the input description.
2.  **Dual-Head MLP:** Two separate Multi-Layer Perceptrons (Linear -> GELU -> LayerNorm -> Linear) project the embeddings into scalar scores for:
    * **Occupation Ranking** (Task A)
    * **Sector/Industry Ranking** (Task B)
3.  **Composite Loss Function:**
    * **Listwise Cross-Entropy:** Optimizes the global probability distribution.
    * **Pairwise Soft Margin:** Optimizes the decision boundary between the correct candidate and the hardest negatives, weighted by prediction uncertainty.

## Features

* **Production-Ready Loop:** Implements standard MLOps patterns including `EarlyStopping`, atomic `CheckpointManager`, and robust JSON logging.
* **Composite Loss:** A custom loss implementation that combines classification and ranking objectives.
* **Dynamic Batching:** A custom Collator handles variable numbers of candidates per example, efficiently padding and masking tensor operations.
* **Observability:** Clean separation of training and validation logic with detailed metric tracking (MRR, Accuracy).

## Installation

```bash
git clone [https://github.com/yourusername/dual-rank-transformer.git](https://github.com/yourusername/dual-rank-transformer.git)
cd dual-rank-transformer
pip install -r requirements.txt
```

## Usage

### Training

The training script leverages `HfArgumentParser`, allowing for flexible configuration via command line arguments.

**Standard Single-GPU Run**
```bash
python train.py \
    --model_name "onlplab/alephbert-base" \
    --batch_size 16 \
    --epochs 10
```

**Distributed Training (Multi-GPU via Accelerate)**
```bash
accelerate launch train.py \
    --mixed_precision fp16 \
    --use_margin True \
    --output_dir "./experiments/run_1"
```

---

## Project Structure

```text
.
├── src/
│   ├── modeling.py   
│   ├── data.py       
│   ├── utils.py      
├── train.py          
├── requirements.txt  
└── README.md
```

---

## Performance & Metrics

The model is evaluated using **Mean Reciprocal Rank (MRR)** and **Top-1 Accuracy**.

* **Listwise Objective:** Ensures the correct label is ranked highly among K candidates.
* **Pairwise Objective:** Ensures a margin separation between positive and negative logits.

---

### Author

**Amir Shibli**

*Data Scientist & Machine Learning Engineer*

linkedin.com/in/amir-shibli
