"""Model builders for Compact CNN1D in FP32 and Quantized QKeras (ATLAS Trigger)."""

from typing import Dict, Optional, Tuple
import tensorflow as tf
import qkeras
from qkeras import QActivation, QConv1D, QDense, quantized_bits, quantized_relu


def build_compact_cnn1d_fp32(
    input_dim: int = 50,
    num_filters: int = 8,
    kernel_size: int = 3,
    dense_units: int = 8,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Builds a simplified compact 1D CNN in FP32 precision.

    Architecture:
        Input(input_dim) -> Reshape -> Conv1D -> ReLU -> MaxPool1D(2) -> Flatten -> Dense -> ReLU -> Dense(1, sigmoid)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="input_rings"),
        tf.keras.layers.Reshape((input_dim, 1), name="reshape_1d"),
        tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_1",
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        tf.keras.layers.Flatten(name="flatten_1"),
        tf.keras.layers.Dense(dense_units, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_dense"),
    ], name="Compact_CNN1D_FP32")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
    act_bits: int = 8,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Builds a simplified compact 1D CNN using QKeras with heterogeneous layer bitwidths.

    Args:
        input_dim: Number of input calorimeter rings (e.g. 50).
        num_filters: Number of filters in QConv1D.
        kernel_size: Size of 1D convolutional kernel.
        dense_units: Hidden units in QDense layer.
        weights_bits: Mapping from layer name to its integer bitwidth (e.g. from KKT allocation).
        act_bits: Quantization bitwidth for activations.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled QKeras Model ready for Quantization-Aware Training (QAT).
    """
    if weights_bits is None:
        weights_bits = {}

    b_conv = weights_bits.get("conv1d_1", 8)
    b_dense = weights_bits.get("dense_1", 8)
    b_out = weights_bits.get("output_dense", 8)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="input_rings"),
        tf.keras.layers.Reshape((input_dim, 1), name="reshape_1d"),
        QConv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_quantizer=quantized_bits(bits=b_conv, integer=2, alpha=1),
            bias_quantizer=quantized_bits(bits=b_conv, integer=2, alpha=1),
            name="conv1d_1",
        ),
        QActivation(activation=quantized_relu(bits=act_bits, integer=2), name="relu_conv1"),
        tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        tf.keras.layers.Flatten(name="flatten_1"),
        QDense(
            units=dense_units,
            kernel_quantizer=quantized_bits(bits=b_dense, integer=2, alpha=1),
            bias_quantizer=quantized_bits(bits=b_dense, integer=2, alpha=1),
            name="dense_1",
        ),
        QActivation(activation=quantized_relu(bits=act_bits, integer=2), name="relu_dense1"),
        QDense(
            units=1,
            kernel_quantizer=quantized_bits(bits=b_out, integer=2, alpha=1),
            bias_quantizer=quantized_bits(bits=b_out, integer=2, alpha=1),
            activation="sigmoid",
            name="output_dense",
        ),
    ], name="Compact_QCNN1D")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
