"""Hardware metrics and Virtex UltraScale+ VU13P resource estimator for FPGA synthesis (ATLAS Trigger)."""

import math
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf

# AMD/Xilinx Virtex UltraScale+ VU13P Official Limits (Table 15 - FPGA Feature Summary)
VU13P_CHIP_LIMITS: Dict[str, float] = {
    "DSP": 12288.0,                                # 12,288 DSP48E2 slices
    "LUT": 1728000.0,                              # 1,728,000 CLB LUTs
    "FF": 3456000.0,                               # 3,456,000 CLB Flip-Flops
    "BRAM_bits": 94.5 * 1024.0 * 1024.0,           # 94.5 Mb Block RAM (2,688 x 36 Kb blocks)
    "URAM_bits": 360.0 * 1024.0 * 1024.0,          # 360.0 Mb UltraRAM (1,280 x 288 Kb blocks)
}

VU13P_LIMITS: Dict[str, float] = {
    "DSP": VU13P_CHIP_LIMITS["DSP"],
    "LUT": VU13P_CHIP_LIMITS["LUT"],
    "FF": VU13P_CHIP_LIMITS["FF"],
    "BRAM_bits": VU13P_CHIP_LIMITS["BRAM_bits"],
}


def get_vu13p_budget(
    scale_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Returns effective hardware resource budgets B_j on the Virtex UltraScale+ VU13P.

    Since a single L0 Ringer tile classifier occupies a tiny fraction of the entire VU13P die
    (which hosts hundreds of parallel trigger towers), `scale_factors` allows specifying the
    maximum per-instance resource allocation (or defaults to 100% of Table 15 limits).

    Args:
        scale_factors: Optional dict mapping resource key ('DSP', 'LUT', 'FF', 'BRAM_bits')
            to a fraction in (0, 1] or explicit limit override.

    Returns:
        Dictionary of resource budgets B_j for {'DSP', 'LUT', 'FF', 'BRAM_bits'}.
    """
    if scale_factors is None:
        return VU13P_LIMITS.copy()

    budgets = {}
    for k, chip_max in VU13P_LIMITS.items():
        factor = scale_factors.get(k, 1.0)
        budgets[k] = float(chip_max * factor) if factor <= 1.0 else float(factor)
    return budgets


def estimate_hardware_resources(
    model_config: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Estimates Virtex UltraScale+ VU13P resource usage (DSP, LUT, FF, BRAM_bits) from layer config.

    Follows UltraScale+ synthesis rules:
    - If w_bits <= 8 and a_bits <= 8: sub-byte arithmetic is mapped onto CLB LUTs:
      LUT_l = macc_ops * (w_bits * a_bits) / 18.0
    - If w_bits > 8 or a_bits > 8: arithmetic uses DSP48E2 slices:
      DSP_l = macc_ops / 64.0 (pipelined initiation interval factor) + control LUTs
    - Flip-Flops (FF) and BRAM storage bits scale with parameter storage (params * w_bits)
      and datapath pipeline registers.

    Args:
        model_config: List of dicts with keys {'w_bits', 'a_bits', 'params', 'macc'}.

    Returns:
        Dict with keys {'DSP', 'LUT', 'FF', 'BRAM_bits'}.
    """
    total_dsp = 0.0
    total_lut = 0.0
    total_ff = 0.0
    total_bits = 0.0

    for layer in model_config:
        w_bits = float(layer.get("w_bits", layer.get("b_w", 8)))
        a_bits = float(layer.get("a_bits", layer.get("b_a", 8)))
        params = float(layer.get("params", 0))
        macc_ops = float(layer.get("macc", layer.get("macs", 0)))

        # UltraScale+ synthesis allocation rule
        if w_bits <= 8 and a_bits <= 8:
            total_lut += macc_ops * (w_bits * a_bits) / 18.0
        else:
            # High-precision (>8 bit) multipliers instantiate DSP48E2 slices
            total_dsp += math.ceil(macc_ops / 64.0)
            total_lut += macc_ops * (w_bits + a_bits) / 36.0

        total_ff += params * w_bits + (macc_ops * a_bits) / 16.0
        total_bits += params * w_bits

    return {
        "DSP": float(total_dsp),
        "LUT": float(total_lut),
        "FF": float(total_ff),
        "BRAM_bits": float(total_bits),
    }


def compute_model_hardware_costs(
    model: tf.keras.Model,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Union[int, Dict[str, int]]] = 8,
    default_act_bits: int = 8,
    default_weight_bits: int = 8,
    input_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute theoretical hardware costs and Virtex UltraScale+ VU13P resources for a model.

    Cost metrics:
    - total_params: Total trainable parameters.
    - total_macs: Total multiply-accumulate operations.
    - total_bops: Bit Operations = sum(MACs_i * b_w,i * b_a,i).
    - total_weight_bits: Total storage bits for model parameters.
    - vu13p_resources: Dict with {'DSP', 'LUT', 'FF', 'BRAM_bits'} estimated usage.

    Args:
        model: Built tf.keras.Model.
        weights_bits: Mapping from layer name to its assigned weight bitwidth w_l.
        act_bits: Global activation bitwidth (int) or per-layer mapping (Dict[str, int]) a_l.
        default_act_bits: Default bitwidth for activations if not specified.
        default_weight_bits: Default bitwidth for weights if not in mapping.
        input_dim: Optional input feature dimension (inferred from model if None).

    Returns:
        Dictionary containing hardware resource estimations and VU13P breakdown.
    """
    if weights_bits is None:
        weights_bits = {}

    if isinstance(act_bits, int):
        act_bits_map: Dict[str, int] = {}
        fallback_act = act_bits
    elif isinstance(act_bits, dict):
        act_bits_map = act_bits
        fallback_act = default_act_bits
    else:
        act_bits_map = {}
        fallback_act = default_act_bits

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
    model_config_for_vu13p = []

    for layer in model.layers:
        layer_cls = layer.__class__.__name__
        if ("Conv1D" in layer_cls or "Dense" in layer_cls) and hasattr(layer, "kernel"):
            out_shape = layer_output_shapes[layer.name]
            kernel = layer.kernel

            if "Conv1D" in layer_cls:
                l_out = int(out_shape[1])
                k_size = int(kernel.shape[0])
                c_in = int(kernel.shape[1])
                c_out = int(kernel.shape[2])
                macs = l_out * k_size * c_in * c_out
                units_or_filters = c_out
            else:
                n_in = int(kernel.shape[0])
                n_out = int(kernel.shape[1])
                macs = n_in * n_out
                units_or_filters = n_out

            b_w = int(weights_bits.get(layer.name, default_weight_bits))
            b_a = int(act_bits_map.get(layer.name, fallback_act))
            bops = macs * b_w * b_a
            num_weights = int(layer.count_params())
            weight_bits = num_weights * b_w

            layer_vu13p = estimate_hardware_resources([{
                "w_bits": b_w,
                "a_bits": b_a,
                "params": num_weights,
                "macc": macs,
            }])

            total_macs += macs
            total_bops += bops
            total_weight_bits += weight_bits

            layer_breakdown.append({
                "layer_name": layer.name,
                "layer_type": layer_cls,
                "units": units_or_filters,
                "macs": macs,
                "b_w": b_w,
                "b_a": b_a,
                "bops": bops,
                "params": num_weights,
                "DSP": layer_vu13p["DSP"],
                "LUT": layer_vu13p["LUT"],
                "FF": layer_vu13p["FF"],
                "BRAM_bits": layer_vu13p["BRAM_bits"],
            })

            model_config_for_vu13p.append({
                "layer_name": layer.name,
                "w_bits": b_w,
                "a_bits": b_a,
                "params": num_weights,
                "macc": macs,
            })

    vu13p_resources = estimate_hardware_resources(model_config_for_vu13p)

    return {
        "total_macs": total_macs,
        "total_bops": total_bops,
        "total_weight_bits": total_weight_bits,
        "total_params": int(model.count_params()),
        "vu13p_resources": vu13p_resources,
        "breakdown": layer_breakdown,
    }
