---
name: keras-tuner-opt
description: >-
  Optimizes hyperparameters of Keras and QKeras models using Keras Tuner. Use
  when the user asks to run hyperparameter tuning, grid search, or Bayesian
  optimization on their neural network.
---

# Keras Tuner Optimizer

This skill structures hyperparameter search spaces for neural networks (especially QKeras models) to find the best balance between accuracy and hardware constraints.

## When to Use

- Setting up a `build_model(hp)` function.
- Running Keras Tuner (`Hyperband`, `RandomSearch`, `BayesianOptimization`).

## Workflow Steps

1. **Wrap the Model**: Create a `def build_model(hp):` function.
2. **Define Search Space**:
   - For layer sizes: `hp.Int('units', min_value=16, max_value=128, step=16)`
   - For quantization bits (crucial for FPGA): `hp.Choice('bits', values=[4, 6, 8])`
   - For learning rate: `hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')`
3. **Instantiate Tuner**: Prefer `kt.BayesianOptimization` or `kt.Hyperband` for efficient search.
4. **Hardware Constraint Tip**: When optimizing QKeras models, track the total bits or parameter sizes to penalize overly large models during custom metric evaluation if necessary.

## Gotchas

- Ensure QKeras quantizers receive the integer parameters correctly from the `hp` object.
- Keras Tuner saves trials to disk; always set a clear `project_name` and directory.
