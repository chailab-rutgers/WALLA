# WALLA: Decentralized LLM aggregation via wagering mechanisms

This repository contains code and experiment configs for **decentralized aggregation of probabilistic predictions from heterogeneous LLMs**, using **wagering mechanisms**.

## Introduction

Having different large language models—each with distinct capabilities, domain expertise, or access to private tools and data—collaborate on complex tasks is increasingly common and valuable. When no single model dominates across tasks and domains, a natural objective is to combine complementary strengths so the collective outperforms any individual contributor. For prediction tasks such as forecasting and multiple-choice question answering, this amounts to aggregating **probabilistic predictions**.

A growing body of work studies how to perform such aggregation. Common approaches (majority voting, self-consistency, multi-agent debate) treat models equally, implicitly assuming similar competence across participants. Training-free weighting heuristics (self-reported confidence, input-token perplexity) can be unreliable when models are miscalibrated and are susceptible to strategic behavior (e.g., inflated reported confidence to gain disproportionate influence). Learning-based methods (routers, stacked generalization) can learn effective domain-dependent weights, but often assume centralized access to model outputs/hidden states and may require retraining as the model pool evolves. Overall, existing methods offer useful trade-offs but have yet to simultaneously account for heterogeneity in model competence, support decentralized deployment, and ensure robustness to strategic behavior.

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

Python **3.12** is tested. All tooling runs inside the project virtual environment at **`.venv`**.

From the repo root:

```bash
bash setup.sh
source .venv/bin/activate   # shell prompt becomes (WALLA)
.venv/bin/python scripts/verify_setup.py
```

Or manually:

```bash
python3 -m venv --prompt WALLA .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

### Optional: API keys

You can provide secrets via a local **`.api_keys.yaml`** (gitignored; create at the repo root):

```yaml
wandb_api_key: "..."   # optional; wandb login also works
hf_token: "..."        # for gated Hugging Face models
```

The scripts ignore empty values and common placeholders like `"your-...-here"`.

## Running the wagering experiments

Experiments are configured via a single YAML file (see `examples/configs/wagering_training/`). Before running, set `cache_path` and `checkpoint_base_dir` in the config to paths on your machine.

### End-to-end pipeline (recommended)

Runs **(optional) calibration → training → evaluation**.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/wagering_pipeline.py \
  examples/configs/wagering_training/walla_v1_2models_mmlu.yaml
```

### Train only / eval only

```bash
.venv/bin/python scripts/wagering_train.py examples/configs/wagering_training/walla_v1_2models_mmlu.yaml
.venv/bin/python scripts/wagering_eval.py  examples/configs/wagering_training/walla_v1_2models_mmlu.yaml
```

### Caching logits, hidden states, and perplexities

Forward passes over the full model pool are expensive. The pipeline caches **per-option log-probs**, **last-token hidden states** (final transformer layer), **labels**, and **prompt perplexities** on disk so later runs can skip inference.

Set `cache_path` in your config. Artifacts are written under:

```
<cache_path>/wagering_model_logits_states_caches/
```

Each file is a compressed `.npz` named like `logits__<model>__<dataset>__<variant>__<hash>.npz` (or `ppl__...` for perplexities). The trailing `<hash>` is an MD5 of the full cache key (see below).

#### Warm the cache before parallel repeat runs

When running many repeats in parallel (e.g. `scripts/wagering_pipeline_repeat.py`), **run `wagering_pipeline.py` once serially first** to populate the cache. Repeat runs only change `shuffle_seed` and checkpoint directories; they reuse the same cached forward-pass outputs. If every parallel worker hits a cache miss at once, each job will load all models and compete for GPU memory and disk writes.

```bash
# 1) Build cache (one GPU, one run)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/wagering_pipeline.py path/to/config.yaml

# 2) Parallel repeats (shared cache_path, separate checkpoint_base_dir per repeat)
.venv/bin/python scripts/wagering_pipeline_repeat.py --config path/to/config.yaml --n-repeats 5 --gpus 0,1,2,3
```

Use the **same `cache_path`** across repeats and across methods that share the same models, datasets, and prompts.

#### Cache key ingredients

**Logits / hidden states / labels** — tuple hashed to pick the file:

| Component | Source |
|-----------|--------|
| `model_key` | Hugging Face model path (`models[].path`). For PubMedQA mixed-context routing, also `::idx=<model_slot>` because the same path can see different prompts per slot. |
| `dataset_key` | `("cfg", schema_version, signature)` where `signature` is an MD5 of the resolved dataset config (HF path, split, `config_name`, and dataset YAML fields). **Not** the run-level `shuffle_seed` unless the dataset block sets `split_seed`. |
| `option_key` | `option_tokens` from the experiment config (e.g. `("A", "B", "C", "D")`). |
| `prompt_variant` | `"default"` for standard datasets. For PubMedQA mixed-context, encodes context-assignment routing and model slot. |
| namespace | PubMedQA adds `pubmedqa_v2_stable_dataset_split_seed` so mixed-context caches stay distinct. |

**Prompt perplexities** — same `model_key`, `dataset_key`, and `prompt_variant`, plus namespace `prompt_perplexity_v1` (and the PubMedQA namespace when applicable).

#### Other things to watch

- **What is *not* in the key:** training hyperparameters, wagering method, `shuffle_seed`, checkpoint paths, and (for most datasets) which hidden layers the wager head uses. Hidden states are always the **last layer, last token**; changing `hidden_layers` in the method config does not invalidate the cache.
- **What *does* invalidate the cache:** different model path, dataset definition (including `size`, split, or HF config), `option_tokens`, PubMedQA prompt/routing strategy, or `split_seed` in the dataset YAML.
- **Partial entries:** a file may contain logits without hidden states. If a later phase needs hidden states (calibration, trainable wagers), it will load the model and fill in the missing arrays.
- **Tripartition train/val/test views** can share one on-disk source cache; the loader slices rows by view index when reading.
- **Stale formats:** older per-layer or pickle hidden-state caches raise an error — delete the affected `.npz` files and recollect.
- **Disk space:** one file per (model [× slot], dataset view, option set, prompt variant); large multi-model sweeps can grow quickly.

## Notes on “wager training”

The goal of the mechanism is to incentivize models to **recognize comparative advantage and limitations**—participating confidently on questions within their expertise and abstaining elsewhere—rather than to improve base predictive capability from payout signals alone. Fine-tuning on payout signals is fundamentally limited without domain-appropriate training data; learning *when to participate* is often both easier and more valuable than trying to become universally competent.

## License

See `LICENSE.md`.

