---
name: jupyter-ai-agents
description: >-
  Enables interactive statistical experimentation, executing code in Jupyter
  Notebooks, loading accelerator datasets, running Keras test snippets, and
  extracting interactive signal and metric plots.
---

# Jupyter AI Agents (Statistical & Interactive Experimentation)

This skill enables running interactive statistical experiments and managing Jupyter notebooks for exploratory data analysis, signal plotting, and quick verification of Keras/QKeras models.

## When to Use

- Interacting with local Jupyter notebooks (`.ipynb` files) or executing notebook cells.
- Loading accelerator/calorimeter datasets and visualizing signal waveforms or ring features.
- Testing quick Keras/QKeras code snippets and loss curves interactively.
- Producing plots (matplotlib, seaborn, plotly) to inspect signal distributions and training progress.

## Best Practices

1. **Notebook Execution & Inspection**:
   - Inspect existing notebooks using CLI or python scripts (`nbformat`, `jupyter nbconvert`, or headless execution via `papermill` / python script).
   - Keep cells modular: one for imports/configuration, one for data loading, one for model definition, and one for visualization.
2. **Signal & Data Visualization**:
   - For accelerator datasets (e.g., energy deposits across calorimeter rings/cells), plot pulse shapes, energy distributions, and feature correlations.
   - Always label axes with appropriate physical units (e.g., sample time / clock cycles, energy in MeV/GeV, ring indices).
3. **Reproducible Experimentation**:
   - Set random seeds for NumPy and TensorFlow when running notebook experiments.
   - Save generated figures (`plt.savefig(...)`) with descriptive names under an artifacts/plots directory for persistent reference.
