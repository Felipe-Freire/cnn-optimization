---
name: qkeras-cnn1d
description: >-
  Develops 1D Convolutional Neural Networks (CNN1D) using QKeras for FPGA
  deployment. Use when the user asks to build, define, or quantize a 1D CNN for
  hardware like ATLAS triggers.
---

# QKeras CNN1D Builder

This skill assists in designing 1D Convolutional Neural Networks using QKeras, specifically targeted for FPGA environments like the HL-LHC ATLAS trigger system.

## When to Use

- Building baseline CNN1D models for signal processing.
- Replacing Keras FP32 layers with QKeras quantized layers.

## Architecture Rules

- **Strict layer usage**: Never use standard Keras layers (`Conv1D`, `Dense`) for the data path. Always use `QConv1D` and `QDense`.
- **Quantization parameters**: Default to `quantized_bits(bits=8, integer=2, alpha=1)` for weights/biases unless otherwise specified.
- **Activations**: Avoid standard `relu` or `sigmoid`. Use `QActivation` with `quantized_relu(bits=8, integer=2)`.
- **Pooling**: Prefer `AveragePooling1D` or `MaxPooling1D`, but be aware of hardware resource costs.

## Example Structure

```python
from qkeras import QActivation, QConv1D, QDense, quantized_bits, quantized_relu
import tensorflow as tf

model = tf.keras.Sequential()

# Use QConv1D for spatial/temporal signal patterns
model.add(QConv1D(
    filters=16,
    kernel_size=3,
    kernel_quantizer=quantized_bits(8, 2, 1),
    bias_quantizer=quantized_bits(8, 2, 1),
    name="qconv1d_1",
    input_shape=(100, 1)  # Example input shape
))
model.add(QActivation(activation=quantized_relu(8, 2), name="relu1"))
```
