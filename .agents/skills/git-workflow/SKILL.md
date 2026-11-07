---
name: git-workflow
description: >-
  Enforces repository hygiene, analyzes git diffs, generates Conventional
  Commits messages, and organizes pull requests to keep experiment histories and
  hyperparameter tuning traceable.
---

# Git Workflow & Commit Hygiene

This skill ensures repository hygiene, structures atomic commits, enforces Conventional Commits standards, and maintains a clean history for hyperparameter tuning experiments and model development.

## When to Use

- Committing changes, creating branches, or preparing pull requests.
- Documenting hyperparameter tuning experiments and model architecture iterations.
- Analyzing `git status` or `git diff` before committing.

## Conventional Commits Convention

All commits must follow the Conventional Commits specification:

```text
<type>(<scope>): <short description in imperative mood>

[optional body explaining WHY and WHAT changed, including hyperparameter metrics if relevant]

[optional footer(s)]
```

### Allowed Types

- `feat`: A new feature or model architecture component.
- `fix`: A bug fix in training scripts, layer definitions, or evaluation metrics.
- `perf`: A code or model change that improves performance, latency, or throughput (e.g., bitwidth reduction, loop vectorization).
- `experiment`: A hyperparameter tuning or trial iteration (e.g., `experiment(tuner): evaluate 4-bit quantizer on Conv1D`).
- `refactor`: Code changes that neither fix a bug nor add a feature.
- `docs`: Documentation updates, markdown notes, or README changes.
- `test`: Adding or correcting tests.
- `chore`: Maintenance tasks, dependencies, gitignore updates.

## Workflow Steps

1. **Review Diff**:
   - Inspect status: `git status`
   - Inspect changes: `git diff --staged` or `git diff`
   - Ensure no unintended files, caches, large dataset binaries, or trial checkpoints are staged.
2. **Atomic Grouping**:
   - Group related modifications together (e.g., separate model definition from tuning config).
3. **Generate Commit Message**:
   - Write clear summary adhering to Conventional Commits.
   - For experiment trials, include relevant metrics (e.g., validation accuracy, parameter count, bitwidth).
4. **Pull Requests**:
   - Organize branches with descriptive names (`feat/...`, `experiment/...`, `fix/...`).
   - Include description linking relevant issue/objective and summary of changes.
