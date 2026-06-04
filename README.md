# WALLA: Decentralized LLM aggregation via wagering mechanisms

This repository contains code and experiment configs for **decentralized aggregation of probabilistic predictions from heterogeneous LLMs**, using **wagering mechanisms** as *principled aggregation weights*.

## Introduction

Having different large language models—each with distinct capabilities, domain expertise, or access to private tools and data—collaborate on complex tasks is increasingly common and valuable. When no single model dominates across tasks and domains, a natural objective is to combine complementary strengths so the collective outperforms any individual contributor. For prediction tasks such as forecasting and multiple-choice question answering, this amounts to aggregating **probabilistic predictions**.

A growing body of work studies how to perform such aggregation. Common approaches (majority voting, self-consistency, multi-agent debate) treat models equally, implicitly assuming similar competence across participants. Training-free weighting heuristics (self-reported confidence, input-token perplexity) can be unreliable when models are miscalibrated and are susceptible to strategic behavior (e.g., inflated reported confidence to gain disproportionate influence). Learning-based methods (routers, stacked generalization) can learn effective domain-dependent weights, but often assume centralized access to model outputs/hidden states and may require retraining as the model pool evolves. Overall, existing methods offer useful trade-offs but have yet to simultaneously account for heterogeneity in model competence, support decentralized deployment, and ensure robustness to strategic behavior. *(TODO: add citations.)*

This repo explores **wagering mechanisms** as a decentralized mechanism-design approach: each agent reports a prediction and chooses a **wager** (stake). The mechanism pays out based on relative performance, scaled by the wager. Interpreting wagers as aggregation weights allows the mechanism to both incentivize truthful prediction and elicit which models should meaningfully contribute.

We implement and study a family of **advantage-aligned wagering mechanisms** that modify the payout so that:
- **predictions remain incentive compatible** (proper-scoring-rule optimum) under general belief structures, and
- **the best-response wager aligns with expected score advantage** relative to a baseline that depends on opponents’ predictions and wagers,
while also yielding a learning signal suitable for gradient-based optimization.

## Repo layout

- **`wagering/`**: wagering methods, aggregation rules, calibration, training, inference utilities.
- **`scripts/`**: runnable entrypoints.
  - `scripts/wagering_pipeline.py`: end-to-end pipeline (calibration → training → evaluation).
  - `scripts/wagering_train.py`: training-only.
  - `scripts/wagering_eval.py`: evaluation-only.
- **`examples/configs/wagering_training/`**: YAML configs for experiments (models/datasets/methods).
- **`artifacts/`**: plots and derived outputs.

## Setup

This repo’s Python tooling is expected to run inside the project virtual environment at **`.venv`**.

```bash
cd /common/home/yl2310/MultiLLMs
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Optional: API keys

You can provide secrets via a local **`.api_keys.yaml`** (not committed):

```yaml
wandb_api_key: "..."
openai_api_key: "..."
```

The scripts ignore empty values and common placeholders like `"your-...-here"`.

## Running the wagering experiments

Experiments are configured via a single YAML file (see `examples/configs/wagering_training/`).

### End-to-end pipeline (recommended)

Runs **(optional) calibration → training → evaluation**.

```bash
cd /common/home/yl2310/MultiLLMs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/wagering_pipeline.py \
  examples/configs/wagering_training/mse_br_wagers_v2_8models.yaml
```

### Train only / eval only

```bash
cd /common/home/yl2310/MultiLLMs
.venv/bin/python scripts/wagering_train.py examples/configs/wagering_training/mse_br_wagers_v2_8models.yaml
.venv/bin/python scripts/wagering_eval.py  examples/configs/wagering_training/mse_br_wagers_v2_8models.yaml
```

### Two-phase (distribution shift) pipeline

If a config defines `phase_shift.phase1` and `phase_shift.phase2`, run:

```bash
cd /common/home/yl2310/MultiLLMs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/wagering_two_phase_pipeline.py \
  path/to/two_phase_config.yaml
```

## Notes on “wager training”

The goal of the mechanism is to incentivize models to **recognize comparative advantage and limitations**—participating confidently on questions within their expertise and abstaining elsewhere—rather than to improve base predictive capability from payout signals alone. Fine-tuning on payout signals is fundamentally limited without domain-appropriate training data; learning *when to participate* is often both easier and more valuable than trying to become universally competent.

## License

See `LICENSE.md`.

