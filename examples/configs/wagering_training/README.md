# Wagering Training Configuration Files

This directory contains configuration files for the multi-LLM wagering training and evaluation pipeline.

## Directory Structure

```
wagering_training/
├── models/              # Shared model configurations
│   ├── gemma_2_9b_it.yaml
│   ├── llama3_aloe_8b_alpha.yaml
│   ├── llama3_1_8b_instruct.yaml
│   └── sera_8b.yaml
├── datasets/            # Shared dataset configurations
│   ├── mmlu.yaml
│   ├── medmcqa.yaml
│   └── arc_easy.yaml
├── full_pipeline_mmlu_medmcqa.yaml  # End-to-end pipeline config (recommended)
├── train_mmlu_medmcqa.yaml          # Training config (for separate training)
├── eval_mmlu_medmcqa_arc.yaml        # Evaluation config (for separate evaluation)
├── README.md            # This file
└── USAGE_GUIDE.md       # Detailed usage guide
```

## When to Use Which Config Type

### End-to-End Pipeline Config (Recommended for Most Cases)

Use `full_pipeline_*.yaml` when:
- Running a complete experiment from scratch
- Training and evaluation use the same datasets/models
- You want automatic checkpoint path handling

```bash
python scripts/wagering_pipeline.py full_pipeline_mmlu_medmcqa.yaml
```

### Separate Train/Eval Configs

Use separate `train_*.yaml` and `eval_*.yaml` when:
- **Re-evaluating with different datasets** (without retraining)
- **Using pre-trained checkpoints** from other runs
- **Debugging** individual phases
- **Different evaluation configs** for the same training run

```bash
# Train once
python scripts/wagering_train.py train_mmlu_medmcqa.yaml

# Evaluate multiple times with different configs
python scripts/wagering_eval.py eval_config1.yaml --checkpoint-path-override /path/to/checkpoint
python scripts/wagering_eval.py eval_config2.yaml --checkpoint-path-override /path/to/checkpoint
```

## Configuration Pattern

The config files use an **include pattern** to reduce duplication and ensure consistency:

1. **Shared Model Configs** (`models/`): Define model loading parameters once
2. **Shared Dataset Configs** (`datasets/`): Define dataset parameters once
3. **Main Configs**: Reference shared configs and override only split-specific settings

### Example: Training Config

```yaml
# Include shared model configs
_include_models:
  - models/gemma_2_9b_it.yaml
  - models/llama3_aloe_8b_alpha.yaml

# Include shared dataset configs
_include_datasets:
  - datasets/mmlu.yaml
  - datasets/medmcqa.yaml

# Override only split-specific settings
datasets:
  - train_split: train
    size: 8
  - train_split: train
    size: 8

# Rest of config...
```

### Example: Evaluation Config

```yaml
# Include same models as training (ensures consistency)
_include_models:
  - models/gemma_2_9b_it.yaml
  - models/llama3_aloe_8b_alpha.yaml

# Include test dataset configs
_include_test_datasets:
  - datasets/mmlu.yaml
  - datasets/medmcqa.yaml

# Override split-specific settings
test_datasets:
  - display_name: mmlu_test
    eval_split: test
    size: 8
  - display_name: medmcqa_test
    eval_split: validation
    size: 8

# OOD dataset
_include_ood_dataset: datasets/arc_easy.yaml
ood_dataset:
  display_name: arc_easy_ood
  eval_split: test
  size: 8
```

### Example: Cached-Logit Calibration Config

```yaml
calibrated: false
_include_calibration: calibration/adaptive_temperature_1000samples.yaml
```

The calibration config is loaded once and trains one temperature head per model on cached hidden states and cached option logits. The frozen heads are then reused by training and evaluation, including eval-only methods such as equal_wagers.

## Benefits

1. **No Duplication**: Model and dataset configs defined once
2. **Consistency**: Training and evaluation use same model/dataset configs
3. **Easy Updates**: Change model config once, affects all experiments
4. **Clear Structure**: Split-specific overrides are explicit
5. **Reusable Calibration**: Temperature scaling artifacts are stored separately and reused across methods when the same calibration config is referenced

## Adding New Models

Create a new file in `models/`:

```yaml
# models/my_model.yaml
path: org/model-name
path_to_load_script: model/default_causal.py
type: CausalLM
instruct: true
load_model_args:
  device_map: auto
  max_memory:
    0: "70GiB"
load_tokenizer_args: {}
```

Then reference it in your training/eval configs:

```yaml
_include_models:
  - models/my_model.yaml
```

## Adding New Datasets

Create a new file in `datasets/`:

```yaml
# datasets/my_dataset.yaml
name: ['org/dataset', 'config']
display_name: my_dataset
text_column: input
label_column: output
batch_size: 8
max_prompt_tokens: 1200
load_from_disk: false
trust_remote_code: false
instruct: true
```

Then reference it in your configs:

```yaml
_include_datasets:
  - datasets/my_dataset.yaml
```

## See Also

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed guide on when to use end-to-end vs separate configs
- **[WAGERING_PIPELINE_README.md](../../../WAGERING_PIPELINE_README.md)** - Overall pipeline documentation
- **[END_TO_END_PIPELINE.md](../../../END_TO_END_PIPELINE.md)** - End-to-end pipeline guide
