---
name: simplify
description: >-
  First-pass code reviewer that identifies and eliminates performance
  bottlenecks, redundant operations, and unnecessary complexity in array
  processing loops and data pipelines.
---

# Simplify (First-Pass Performance & Code Reviewer)

This skill acts as a first-pass reviewer to ensure that data transformation loops, model evaluation code, and array manipulations run at maximum efficiency with clean, readable code.

## When to Use

- Reviewing functions that iterate over large datasets, signals, or feature arrays.
- Eliminating Python `for` loops in favor of vectorized NumPy / TensorFlow operations.
- Removing redundant disk IO, duplicate calculations, or memory leaks in data loaders.
- Refactoring convoluted multi-step functions into clean, single-responsibility components.

## Optimization Checklist

1. **Array Vectorization**:
   - Replace explicit Python `for` loops with vectorized NumPy or TensorFlow tensor operations.
   - Use `np.vectorize` only as a last resort; prefer true broadcast operations (`np.where`, slicing, matrix multiplications).
2. **Memory Efficiency**:
   - Avoid creating unnecessary intermediate array copies; use in-place operations (`out=` parameters) or views where safe.
   - Ensure generator pipelines or `tf.data.Dataset.prefetch` are used instead of holding massive uncompressed datasets in RAM.
3. **Complexity Reduction**:
   - Flatten deeply nested conditional branches.
   - Extract pure helper functions for repetitive signal normalization or filtering logic.
   - Remove dead code, redundant queries, and unneeded recalculations within training epochs.
