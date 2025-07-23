# WSD (Warmup-Stable-Decay) Scheduler

## Overview

The WSD (Warmup-Stable-Decay) scheduler is a three-phase learning rate scheduler that provides:

1. **Warmup Phase**: Linear increase from 0 to peak learning rate
2. **Stable Phase**: Constant learning rate at peak value
3. **Decay Phase**: Linear decrease from peak to final learning rate

This scheduler is particularly effective for transformer training and has been shown to improve convergence and final performance.

## Configuration

This implementation uses the official transformers `warmup_stable_decay` scheduler. To use the WSD scheduler, set the following parameters in your YAML configuration file:

```yaml
# Basic learning rate setting
learning_rate: 0.0005  # Peak learning rate (5.0 × 10^-4)

# WSD scheduler configuration
lr_scheduler_type: warmup_stable_decay
warmup_ratio: 0.01      # 2000 steps / 200000 total steps = 0.01
warmup_steps: 2000      # Number of warmup steps (alternative to warmup_ratio)
decay_steps: 20000      # Number of decay steps (10% of total steps)
min_lr_ratio: 0.0       # Final LR as ratio of peak LR (0 = decay to zero)
```

## Parameters

### Required Parameters

- **`lr_scheduler_type`**: Set to `"warmup_stable_decay"` to enable WSD scheduler.

### WSD-Specific Parameters

- **`warmup_steps`** (optional): Number of steps for the warmup phase. Alternative to `warmup_ratio`.
- **`warmup_ratio`** (optional): Fraction of total steps for warmup phase. Alternative to `warmup_steps`.
- **`decay_steps`** (optional): Number of steps for the decay phase.
- **`min_lr_ratio`** (default: 0.0): Final learning rate as a ratio of the peak learning rate. 0.0 means decay to zero, 0.1 means decay to 10% of peak LR.

## Example Usage

### Example 1: Paper Configuration
Based on "We used WSD scheduler with 2000 steps warmup, learning rate 5.0 × 10^-4 and 10% decay":

```yaml
learning_rate: 0.0005
lr_scheduler_type: warmup_stable_decay
warmup_steps: 2000
decay_steps: 20000  # 10% of 200k total steps
min_lr_ratio: 0.0
```

### Example 2: Using Ratios
```yaml
learning_rate: 0.001
lr_scheduler_type: warmup_stable_decay
warmup_ratio: 0.025     # 2.5% of total steps for warmup
min_lr_ratio: 0.1       # Decay to 10% of peak LR
```

## Phase Calculation

For a training run with:
- Total steps: 200,000
- Warmup steps: 2,000
- Decay ratio: 0.1 (10%)

The phases will be:
1. **Warmup**: Steps 1-2,000 (linear increase from 0 to peak LR)
2. **Stable**: Steps 2,001-180,000 (constant at peak LR)
3. **Decay**: Steps 180,001-200,000 (linear decrease from peak to final LR)

## Implementation Details

The WSD scheduler is implemented as:
- `WSDScheduler` class: Core scheduler logic
- `WSDTrainer` class: Custom trainer that integrates the scheduler
- Automatic phase detection and learning rate calculation

## Comparison with Default Schedulers

| Scheduler | Warmup | Stable Phase | Decay |
|-----------|--------|--------------|-------|
| Linear | ✓ | ✗ | ✓ (immediate) |
| Cosine | ✓ | ✗ | ✓ (smooth) |
| WSD | ✓ | ✓ | ✓ (linear) |

The WSD scheduler's stable phase allows the model to train at peak learning rate for most of the training, potentially leading to better convergence.

## Running with WSD

```bash
# Use the example configuration
python scripts/training/train.py --config scripts/training/configs/t5-v1_1-base-wsd-example.yaml

# Or modify your existing config to include WSD parameters
python scripts/training/train.py --config your_config.yaml
```

## Logging

When WSD scheduler is active, you'll see log messages like:
```
Using WSD scheduler with warmup_steps=2000, decay_ratio=0.1, final_lr_ratio=0.0
```

When using default scheduler:
```
Using default scheduler: linear with warmup_ratio=0.0
```
