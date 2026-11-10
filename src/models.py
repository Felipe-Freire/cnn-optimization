"""Model builders for Compact CNN1D in FP32 and Quantized QKeras (ATLAS Trigger).

The quantizer convention lives here (``weight_quantizer``, ``relu_quantizer``, ``input_quantizer``)
and is reused by the hardware cost model and by the sensitivity analysis, so that the bits that are
trained, the bits that are costed and the bits whose error is modeled are exactly the same objects.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from typing import Dict, Optional, Union

import numpy as np
import tf_keras as keras
from qkeras import QActivation, QConv1D, QDense, quantized_bits, quantized_relu

# Layers with weights (in order) and the activations that follow conv1d_1 / dense_1.
COMPUTE_LAYERS = ("conv1d_1", "dense_1", "output_dense")
ACTIVATION_LAYERS = ("conv1d_1", "dense_1")  # key = layer whose (ReLU) output is quantized


def integer_bits(bits: int) -> int:
    """Integer bits (sign excluded) used by every quantizer of the project: min(2, b - 1)."""
    return min(2, max(0, int(bits) - 1))


def weight_quantizer(bits: int) -> quantized_bits:
    """Signed fixed-range quantizer for weights/biases: values k * 2^(I+1-b), k in [-2^(b-1), 2^(b-1)-1]."""
    return quantized_bits(bits=int(bits), integer=integer_bits(bits), alpha=1)


def weight_lsb(bits: int) -> float:
    """Quantization step of ``weight_quantizer(bits)``."""
    return 2.0 ** (integer_bits(bits) + 1 - int(bits))


def relu_quantizer(bits: int) -> quantized_relu:
    """Unsigned ReLU quantizer: values in [0, 2^I) with step 2^(I-b)."""
    return quantized_relu(bits=int(bits), integer=integer_bits(bits))


def input_quantizer(bits: int) -> quantized_bits:
    """Unsigned quantizer for the L1-normalized rings (they live in [0, 1)): step 2^-b."""
    return quantized_bits(bits=int(bits), integer=0, keep_negative=False, alpha=1)


def quantize_weights_int(values: np.ndarray, bits: int) -> np.ndarray:
    """Quantize float weights with the project convention and return the integer codes k (w_q = k * LSB)."""
    wq = np.asarray(weight_quantizer(bits)(np.asarray(values, dtype=np.float32)))
    return np.rint(wq / weight_lsb(bits)).astype(np.int64)


def build_compact_cnn1d_fp32(
    input_dim: int = 50,
    num_filters: int = 8,
    kernel_size: int = 3,
    dense_units: int = 8,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Builds a simplified compact 1D CNN in FP32 precision.

    Architecture:
        Input(input_dim) -> Reshape -> Conv1D -> ReLU -> MaxPool1D(2) -> Flatten -> Dense -> ReLU -> Dense(1, sigmoid)
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,), name="input_rings"),
        keras.layers.Reshape((input_dim, 1), name="reshape_1d"),
        keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_1",
        ),
        keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        keras.layers.Flatten(name="flatten_1"),
        keras.layers.Dense(dense_units, activation="relu", name="dense_1"),
        keras.layers.Dense(1, activation="sigmoid", name="output_dense"),
    ], name="Compact_CNN1D_FP32")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_compact_qcnn1d(
    input_dim: int = 50,
    num_filters: int = 8,
    kernel_size: int = 3,
    dense_units: int = 8,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Union[int, Dict[str, int]] = 8,
    input_bits: Optional[int] = None,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Builds a simplified compact 1D CNN using QKeras with heterogeneous layer bitwidths (w_l, a_l).

    Args:
        input_dim: Number of input calorimeter rings (e.g. 25, 50, or 100).
        num_filters: Number of filters in QConv1D (n_1).
        kernel_size: Size of 1D convolutional kernel.
        dense_units: Hidden units in QDense layer (n_2).
        weights_bits: Mapping from layer name to its integer weight bitwidth w_l.
        act_bits: Global activation bitwidth (int) or per-layer dict {'conv1d_1': a_1, 'dense_1': a_2},
            where the key is the layer whose ReLU output is quantized.
        input_bits: If given, quantizes the rings entering the network to ``ufixed<input_bits, 0>``
            (the format they must have at the FPGA input). ``None`` keeps a float input.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled QKeras Model ready for Quantization-Aware Training (QAT).
    """
    if weights_bits is None:
        weights_bits = {}

    b_conv = int(weights_bits.get("conv1d_1", 8))
    b_dense = int(weights_bits.get("dense_1", 8))
    b_out = int(weights_bits.get("output_dense", 8))

    if isinstance(act_bits, dict):
        a_conv = int(act_bits.get("conv1d_1", 8))
        a_dense = int(act_bits.get("dense_1", 8))
    else:
        a_conv = int(act_bits)
        a_dense = int(act_bits)

    layers = [keras.layers.Input(shape=(input_dim,), name="input_rings")]
    if input_bits is not None:
        layers.append(QActivation(input_quantizer(input_bits), name="input_quant"))
    layers += [
        keras.layers.Reshape((input_dim, 1), name="reshape_1d"),
        QConv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_quantizer=weight_quantizer(b_conv),
            bias_quantizer=weight_quantizer(b_conv),
            name="conv1d_1",
        ),
        QActivation(activation=relu_quantizer(a_conv), name="relu_conv1"),
        keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        keras.layers.Flatten(name="flatten_1"),
        QDense(
            units=dense_units,
            kernel_quantizer=weight_quantizer(b_dense),
            bias_quantizer=weight_quantizer(b_dense),
            name="dense_1",
        ),
        QActivation(activation=relu_quantizer(a_dense), name="relu_dense1"),
        QDense(
            units=1,
            kernel_quantizer=weight_quantizer(b_out),
            bias_quantizer=weight_quantizer(b_out),
            activation="sigmoid",
            name="output_dense",
        ),
    ]
    model = keras.Sequential(layers, name="Compact_QCNN1D")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def transfer_weights_to_qmodel(
    source_model: keras.Model, target_qmodel: keras.Model
) -> None:
    """Transfers learned weights/biases from an FP32 reference model to a QKeras model (Warm-Start QAT).

    Matches layers by name ('conv1d_1', 'dense_1', 'output_dense') when shapes are compatible.
    """
    source_layers = {l.name: l for l in source_model.layers}
    for t_layer in target_qmodel.layers:
        if t_layer.name in source_layers:
            s_layer = source_layers[t_layer.name]
            s_weights = s_layer.get_weights()
            t_weights = t_layer.get_weights()
            if len(s_weights) == len(t_weights) and all(
                sw.shape == tw.shape for sw, tw in zip(s_weights, t_weights)
            ):
                t_layer.set_weights(s_weights)
