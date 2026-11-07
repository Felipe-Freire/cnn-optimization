"""Hardware metrics and cost estimator for FPGA synthesis (ATLAS Trigger)."""

from typing import Dict, List, Optional, Tuple
import tensorflow as tf


def compute_model_hardware_costs(
    model: tf.keras.Model,
    weights_bits: Optional[Dict[str, int]] = None,
    default_act_bits: int = 8,
    default_weight_bits: int = 8,
    input_dim: Optional[int] = None,
) -> Dict[str, any]:
    """Compute theoretical hardware costs for a given network architecture.

    Cost metrics:
    - total_params: Total trainable parameters.
    - total_macs: Total multiply-accumulate operations.
    - total_bops: Bit Operations = sum(MACs_i * b_w,i * b_a,i).
    - total_weight_bits: Total storage bits for model parameters.

    Args:
        model: Built tf.keras.Model.
        weights_bits: Mapping from layer name to its assigned weight bitwidth.
        default_act_bits: Default bitwidth for activations (inputs to layers).
        default_weight_bits: Default bitwidth for weights if not in mapping.
        input_dim: Optional input feature dimension (inferred from model if None).

    Returns:
        Dictionary containing hardware resource estimations.
    """
    if weights_bits is None:
        weights_bits = {}

    # Infer input dim from model input shape if not provided
    if input_dim is None:
        in_shape = model.input_shape
        input_dim = in_shape[-1] if in_shape is not None else 50

    # Trace layer output shapes using a dummy tensor
    dummy_x = tf.zeros((1, input_dim))
    curr_tensor = dummy_x
    layer_output_shapes = {}

    for layer in model.layers:
        curr_tensor = layer(curr_tensor)
        layer_output_shapes[layer.name] = curr_tensor.shape

    total_macs = 0
    total_bops = 0
    total_weight_bits = 0
    layer_breakdown = []

    for layer in model.layers:
        layer_cls = layer.__class__.__name__
        if ("Conv1D" in layer_cls or "Dense" in layer_cls) and hasattr(layer, "kernel"):
            out_shape = layer_output_shapes[layer.name]
            kernel = layer.kernel

            if "Conv1D" in layer_cls:
                # out_shape: (batch, L_out, C_out)
                # kernel: (kernel_size, C_in, C_out)
                l_out = int(out_shape[1])
                k_size = int(kernel.shape[0])
                c_in = int(kernel.shape[1])
                c_out = int(kernel.shape[2])
                macs = l_out * k_size * c_in * c_out
            else:
                # Dense kernel: (n_in, n_out)
                n_in = int(kernel.shape[0])
                n_out = int(kernel.shape[1])
                macs = n_in * n_out

            b_w = weights_bits.get(layer.name, default_weight_bits)
            b_a = default_act_bits
            bops = macs * b_w * b_a
            num_weights = layer.count_params()
            weight_bits = num_weights * b_w

            total_macs += macs
            total_bops += bops
            total_weight_bits += weight_bits

            layer_breakdown.append({
                "layer_name": layer.name,
                "layer_type": layer_cls,
                "macs": macs,
                "b_w": b_w,
                "b_a": b_a,
                "bops": bops,
                "params": num_weights,
            })

    return {
        "total_macs": total_macs,
        "total_bops": total_bops,
        "total_weight_bits": total_weight_bits,
        "total_params": model.count_params(),
        "breakdown": layer_breakdown,
    }
