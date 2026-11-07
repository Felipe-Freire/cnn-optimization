---
name: python-backend
description: >-
  Enforces clean Python architecture, modular project organization, strict type
  annotations, and Test-Driven Development (TDD) for TensorFlow and QKeras signal
  processing pipelines.
---

# Python Backend & Architecture Standards

This skill prevents unstructured scripting and ensures that model training, data ingestion, and preprocessing code adhere to clean architecture principles, strict type safety, and Test-Driven Development (TDD).

## When to Use

- Structuring or refactoring Python codebases for machine learning.
- Preparing TensorFlow / QKeras models for physical hardware deployment (e.g., FPGA / HLS synthesis).
- Writing unit tests, data validation fixtures, and pipeline integration tests.
- Transitioning one-off notebook scripts into reusable, tested production modules.

## Architecture Standards

1. **Modular Separation**:
   - `data/`: Dataset loaders, validation schemas, and preprocessing pipelines.
   - `models/`: Model architecture definitions, layer builders, and quantization specs.
   - `training/`: Training loops, callback configs, and tuner setups.
   - `utils/`: Metric calculations, visualization helpers, and hardware conversion scripts.
2. **Type Annotations & Validation**:
   - Use Python type hints (`typing`, `numpy.typing.NDArray`) across all function signatures.
   - Validate array shapes and tensor dimensions at pipeline boundaries.
3. **Test-Driven Development (TDD)**:
   - Write unit tests (`pytest`) before or alongside implementation.
   - Test data shapes, quantization bounds, model output shapes, and gradient flow on synthetic dummy data.
   - Ensure tests run fast and deterministically.
