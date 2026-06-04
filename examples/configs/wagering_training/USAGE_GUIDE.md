# Wagering Training Config Usage Guide

## Quick Start

### Option 1: End-to-End Pipeline (Recommended)

For most use cases, use the full pipeline config:

```bash
python scripts/wagering_pipeline.py full_pipeline_mmlu_medmcqa.yaml
```

This automatically:
- Runs training
- Saves checkpoint with unique name
- Runs evaluation with the trained checkpoint
- No manual path updates needed

### Option 2: Separate Training and Evaluation

Use separate configs when you need more control:

```bash
# Step 1: Train
python scripts/wagering_train.py train_mmlu_medmcqa.yaml
# Checkpoint saved to: /common/users/yl2310/MultiLLMs/checkpoints/{unique_dir}/final

# Step 2: Evaluate (update checkpoint_path in eval config first)
python scripts/wagering_eval.py eval_mmlu_medmcqa_arc.yaml
```

## When to Use Each Approach

### Use End-to-End Pipeline When:
- ✅ Running a complete experiment from scratch
- ✅ Training and evaluation use the same models/datasets
- ✅ You want automatic checkpoint path handling
- ✅ Standard workflow

### Use Separate Configs When:
- ✅ **Re-evaluating with different datasets** (train once, evaluate many times)
- ✅ **Using pre-trained checkpoints** from other experiments
- ✅ **Debugging** - need to test training or evaluation separately
- ✅ **Different evaluation configs** for the same training run
- ✅ **Iterative development** - faster iteration on one phase

## Example Workflows

### Workflow 1: Train Once, Evaluate Multiple Times

```bash
# Train once
python scripts/wagering_train.py train_mmlu_medmcqa.yaml
# Checkpoint: /common/users/yl2310/MultiLLMs/checkpoints/models_..._hash/final

# Evaluate on different datasets
python scripts/wagering_eval.py eval_mmlu_test.yaml \
    --checkpoint-path-override /common/users/yl2310/MultiLLMs/checkpoints/models_..._hash/final

python scripts/wagering_eval.py eval_medmcqa_test.yaml \
    --checkpoint-path-override /common/users/yl2310/MultiLLMs/checkpoints/models_..._hash/final
```

### Workflow 2: Using Pre-trained Checkpoint

```bash
# Use checkpoint from a previous experiment
python scripts/wagering_eval.py eval_config.yaml \
    --checkpoint-path-override /common/users/yl2310/MultiLLMs/checkpoints/models_..._hash/final
```

### Workflow 3: Debugging

```bash
# Test training only
python scripts/wagering_train.py train_config.yaml

# Test evaluation only (with existing checkpoint)
python scripts/wagering_eval.py eval_config.yaml \
    --checkpoint-path-override /path/to/checkpoint
```

## Available Config Files

### Full Pipeline Configs
- `full_pipeline_mmlu_medmcqa.yaml` - Complete end-to-end example

### Training Configs
- `train_mmlu_medmcqa.yaml` - Training config for MMLU + MedMCQA

### Evaluation Configs
- `eval_mmlu_medmcqa_arc.yaml` - Evaluation config with MMLU, MedMCQA, and ARC-Easy OOD

## Checkpoint Directory Naming

Checkpoints are automatically named based on:
- Models used
- Datasets used
- Wagering method
- Aggregation method
- Hash for uniqueness

Example:
```
/common/users/yl2310/MultiLLMs/checkpoints/
  models_HPAI-BSC_Llama3-Aloe-8B-Alpha_google_gemma-2-9b-it_datasets_medmcqa_mmlu_wagering_equal_wagers_agg_weighted_linear_0b7d3793/
    final/
      wagering_state.pt
```

This ensures:
- ✅ No conflicts between different runs
- ✅ Easy to identify what each checkpoint contains
- ✅ Can have multiple checkpoints for different configurations

