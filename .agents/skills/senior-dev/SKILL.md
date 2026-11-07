---
name: senior-dev
description: >-
  Orchestrates the software development lifecycle (SDLC), acting as Product
  Analyst and System Architect to design neural network architectures, structure
  data pipelines, and plan execution steps before generating code.
---

# Senior Dev SDLC Orchestrator & Architect

This skill provides senior-level planning, architectural design, and lifecycle orchestration before jumping directly into implementation. It coordinates requirements analysis, pipeline design, and execution phases.

## When to Use

- Planning a new feature, model iteration, or refactor.
- Designing the data ingestion, preprocessing, and training pipeline before writing Keras code.
- Defining the neural network architecture topology and matching it against system constraints.
- Breaking down complex modeling tasks into ordered, verifiable phases.

## Perspectives & Roles

### 1. Product Analyst Mode
- **Requirements Clarification**: Identify business/scientific objectives (e.g., ATLAS trigger latency, trigger efficiency, false positive rate).
- **Constraint Definition**: Collect hardware targets (FPGA resources, DSP slices, LUT limits, memory footprint) and dataset formats.
- **Success Criteria**: Define quantifiable metrics for model convergence and hardware feasibility.

### 2. Software & ML Architect Mode
- **Pipeline Architecture**: Map out data loading (e.g., `tf.data.Dataset`, generator vs in-memory), normalizations, and train/val/test splits.
- **Model Topology**: Sketch the network architecture (layer dimensions, receptive fields, quantization strategy) and evaluate compatibility with downstream toolchains (e.g., hls4ml).
- **Module Boundaries**: Separate dataset IO, model definition, training/tuning loops, and export/evaluation utilities.

## Workflow

1. **Analyze Requirements**: Validate inputs, constraints, and baseline expectations.
2. **Draft Architecture Spec**: Detail data schemas, shapes, layer types, and hyperparameters.
3. **Task Breakdown**: Create a sequenced checklist of implementation tasks (data pipeline -> baseline model -> quantized model -> tuning -> export).
4. **Handoff to Implementation**: Once aligned, proceed with disciplined code generation.
