"""Hardware metrics and Virtex UltraScale+ VU13P resource estimator with qkeras.qtools integration.

Resource model (DSP, LUT, FF, BRAM)
-----------------------------------
In hls4ml ``io_parallel`` / ``Latency`` / ``ReuseFactor=1`` the weights are constants wired into the
logic, so the cost of a layer is set by the adder trees that implement its constant multiplications,
not by a generic w x a multiplier:

* a zero weight costs nothing and a power-of-two weight is a shift (wiring);
* a constant with d nonzero canonical-signed-digit (CSD) digits costs (d - 1) adders, unless it is
  wide (max(w, a_in) > 10 bits) and has >= 3 digits, in which case HLS maps it to one DSP48E2;
* the accumulators feeding each ReLU (+ max-pooling) add logic that does not shrink with the trees;
* weights are never stored in BRAM: the only RAM block is the sigmoid table (1 RAMB18).

Per layer l (positions P_l computed in parallel, fan-in n_l, integer weight codes k):

    adders_l     = P_l * sum_o [ (terms_o - 1) + sum_{k in o, LUT} (csd(k) - 1) ]
    adder_bits_l = adders_l * (a_in,l + w_l + ceil(log2 n_l) / 2)
    relu_bits_l  = P_l * n_out,l * (a_in,l + w_l + ceil(log2 n_l))   if a ReLU follows l
    DSP_l        = P_l * #{k : max(w_l, a_in,l) > 10 and csd(k) >= 3}

    LUT = beta_LUT * sum adder_bits + gamma_LUT * sum relu_bits
    FF  = beta_FF  * sum adder_bits + delta_FF * DSP + gamma_FF

The coefficients are fitted by least squares on post-route utilization of this network synthesized
with hls4ml -> Vitis HLS -> Vivado (src/calibration/vu13p_hls4ml_synthesis.json, produced by the
cnn-fpga-research repository). ``CostModel.calibration_table`` reports the fit and leave-one-out
errors; ``CostModel.margins`` turns the worst leave-one-out under-prediction of each resource into a
safety margin for the budgets. Validity domain: io_parallel, Latency, ReuseFactor = 1, Conv1D with
ParallelizationFactor = output width, VU13P at 240 MHz, this topology.

``estimate_hardware_resources`` keeps the MAC-based analytical estimate (LUT = MAC w a / 18, ...),
which does not see weight values or adder trees and under-predicts synthesized LUTs by 3x-24x;
decisions should use ``calibrated_resources`` / ``make_cost_fn``.
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
import qkeras
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings

from src.models import quantizer_integer_bits

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


RAMB18_BITS = 18 * 1024
CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration" / "vu13p_hls4ml_synthesis.json"


def get_vu13p_budget(
    scale_factors: Optional[Dict[str, float]] = None,
    margins: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Returns effective hardware resource budgets B_j on the Virtex UltraScale+ VU13P.

    Since a single L0 Ringer tile classifier occupies a tiny fraction of the entire VU13P die
    (which hosts hundreds of parallel trigger towers), `scale_factors` allows specifying the
    maximum per-instance resource allocation (or defaults to 100% of Table 15 limits).

    Args:
        scale_factors: Optional dict mapping resource key ('DSP', 'LUT', 'FF', 'BRAM_bits')
            to a fraction in (0, 1] or explicit limit override.
        margins: Optional relative safety margin per resource (e.g. CostModel.margins); the
            budget becomes B_j / (1 + m_j), so that a configuration the cost model places at the
            limit still fits if the model under-predicts by m_j.

    Returns:
        Dictionary of resource budgets B_j for {'DSP', 'LUT', 'FF', 'BRAM_bits'}.
    """
    budgets = {}
    for k, chip_max in VU13P_LIMITS.items():
        factor = 1.0 if scale_factors is None else scale_factors.get(k, 1.0)
        budgets[k] = float(chip_max * factor) if factor <= 1.0 else float(factor)
    if margins:
        budgets = {k: v / (1.0 + float(margins.get(k, 0.0))) for k, v in budgets.items()}
    return budgets


def layer_input_bits(
    layer_names: List[str],
    act_bits: Dict[str, int],
    input_bits: int = 8,
    default_act_bits: int = 8,
) -> Dict[str, int]:
    """Precision of the tensor that ENTERS each compute layer, in data-path order.

    The multiplications of a layer use the precision of its input: the network input for the
    first layer, the quantized ReLU output of the previous layer for the others. ``act_bits`` is
    keyed by the layer whose output is quantized ({"conv1d_1": a_1, "dense_1": a_2}), so
    a_in(conv1d_1) = input_bits, a_in(dense_1) = a_1 and a_in(output_dense) = a_2.
    """
    a_in: Dict[str, int] = {}
    current = int(input_bits)
    for name in layer_names:
        a_in[name] = current
        current = int(act_bits.get(name, default_act_bits))
    return a_in


def _compute_layer_names(model: tf.keras.Model) -> List[str]:
    return [
        layer.name for layer in model.layers
        if ("Conv1D" in layer.__class__.__name__ or "Dense" in layer.__class__.__name__)
        and hasattr(layer, "kernel")
    ]


def csd_digits(n: int) -> int:
    """Nonzero digits of |n| in canonical signed-digit form (0 for n = 0, 1 for powers of two)."""
    n = abs(int(n))
    count = 0
    while n:
        if n & 1:
            n -= 2 - (n & 3)  # digit +1 or -1
            count += 1
        n >>= 1
    return count


def quantize_weights_int(values: np.ndarray, bits: int) -> np.ndarray:
    """Integer codes k of quantized_bits(bits, I, alpha=1) applied to float weights (w_q = k * LSB)."""
    integer = quantizer_integer_bits(bits)
    lsb = 2.0 ** (integer + 1 - int(bits))
    quantized = np.asarray(qkeras.quantized_bits(int(bits), integer, alpha=1)(np.asarray(values, dtype=np.float32)))
    return np.rint(quantized / lsb).astype(np.int64)


@dataclass
class LayerSpec:
    """What the structural cost model needs about one compute layer."""

    name: str
    kernel_int: np.ndarray  # integer weight codes, shape (fan_in, n_out)
    bias_int: np.ndarray    # integer bias codes, shape (n_out,)
    w_bits: int
    a_in_bits: int
    positions: int = 1      # output positions computed in parallel (Conv1D: output width)
    relu_after: bool = False

    @property
    def fan_in(self) -> int:
        return int(self.kernel_int.shape[0])


def layer_features(spec: LayerSpec, dsp_min_bits: int = 10, dsp_min_csd: int = 3) -> Dict[str, float]:
    """Adders, adder bits, ReLU input bits and DSPs of one layer (see module docstring)."""
    k = np.asarray(spec.kernel_int)
    b = np.asarray(spec.bias_int).ravel()
    wide = max(spec.w_bits, spec.a_in_bits) > dsp_min_bits
    csd = np.vectorize(csd_digits, otypes=[np.int64])(k) if k.size else k
    nonzero = k != 0
    dsp_mask = nonzero & wide & (csd >= dsp_min_csd)
    extra_adders = np.where(nonzero & ~dsp_mask, csd - 1, 0)
    terms = nonzero.sum(axis=0) + (b != 0).astype(np.int64)
    adders = float(np.maximum(terms - 1, 0).sum() + extra_adders.sum()) * spec.positions
    log_fan_in = math.ceil(math.log2(max(spec.fan_in, 1)))
    width = spec.a_in_bits + spec.w_bits + log_fan_in / 2.0
    acc_width = spec.a_in_bits + spec.w_bits + log_fan_in
    return {
        "adders": adders,
        "adder_bits": adders * width,
        "relu_bits": float(spec.positions * k.shape[1] * acc_width) if spec.relu_after else 0.0,
        "DSP": float(dsp_mask.sum() * spec.positions),
        "nonzero_frac": float(nonzero.mean()) if k.size else 0.0,
    }


@dataclass
class CostModel:
    """Structural VU13P cost model with coefficients fitted on synthesized points."""

    beta_lut: float
    gamma_lut: float
    beta_ff: float
    delta_ff: float
    gamma_ff: float
    ramb18: float = 1.0
    dsp_min_bits: int = 10
    dsp_min_csd: int = 3
    calibration: List[Dict[str, Any]] = field(default_factory=list)

    def layer_resources(self, spec: LayerSpec) -> Dict[str, float]:
        f = layer_features(spec, self.dsp_min_bits, self.dsp_min_csd)
        return {
            "DSP": f["DSP"],
            "LUT": self.beta_lut * f["adder_bits"] + self.gamma_lut * f["relu_bits"],
            "FF": self.beta_ff * f["adder_bits"] + self.delta_ff * f["DSP"],
            "adder_bits": f["adder_bits"],
            "relu_bits": f["relu_bits"],
            "nonzero_frac": f["nonzero_frac"],
        }

    def estimate(self, specs: Sequence[LayerSpec]) -> Dict[str, float]:
        """Totals: DSP, LUT, FF, RAMB18 and BRAM_bits (= RAMB18 x 18 Kb), plus the features."""
        total = {"DSP": 0.0, "LUT": 0.0, "FF": self.gamma_ff, "adder_bits": 0.0, "relu_bits": 0.0}
        for spec in specs:
            res = self.layer_resources(spec)
            for key in ("DSP", "LUT", "FF", "adder_bits", "relu_bits"):
                total[key] += res[key]
        total["RAMB18"] = self.ramb18
        total["BRAM_bits"] = self.ramb18 * RAMB18_BITS
        return total

    @property
    def margins(self) -> Dict[str, float]:
        """Worst leave-one-out under-prediction per resource (fraction), for get_vu13p_budget."""
        out = {}
        for key in ("DSP", "LUT", "FF"):
            errs = [c["loo_rel_err"][key] for c in self.calibration if c["in_domain"] and c["loo_rel_err"]]
            out[key] = float(max([0.0] + [-e for e in errs if np.isfinite(e)]))
        return out

    @staticmethod
    def _design(rows):
        adder_bits = np.array([r["adder_bits"] for r in rows])
        relu_bits = np.array([r["relu_bits"] for r in rows])
        dsp = np.array([r["DSP_pred"] for r in rows])
        return np.c_[adder_bits, relu_bits], np.c_[adder_bits, dsp, np.ones_like(adder_bits)]

    @classmethod
    def from_calibration(cls, path: Union[str, Path] = CALIBRATION_PATH, dsp_min_bits: int = 10,
                         dsp_min_csd: int = 3) -> "CostModel":
        """Least-squares fit on the in-domain points (Conv1D fully parallel), with leave-one-out errors."""
        points = json.loads(Path(path).read_text())["points"]
        rows = []
        for p in points:
            specs = [
                LayerSpec(name, np.asarray(l["kernel_int"]), np.asarray(l["bias_int"]), l["w_bits"],
                          l["a_in_bits"], l["positions_parallel"], l.get("relu_after", False))
                for name, l in p["layers"].items()
            ]
            feats = [layer_features(s, dsp_min_bits, dsp_min_csd) for s in specs]
            rows.append({
                "name": p["name"],
                "in_domain": p["conv_parallelization_factor"] > 1,
                "adder_bits": sum(f["adder_bits"] for f in feats),
                "relu_bits": sum(f["relu_bits"] for f in feats),
                "DSP_pred": sum(f["DSP"] for f in feats),
                "measured": {k: float(p["post_route"][k]) for k in ("DSP", "LUT", "FF")},
            })
        fit = [r for r in rows if r["in_domain"]]

        def solve(rows_):
            x_lut, x_ff = cls._design(rows_)
            c_lut = np.linalg.lstsq(x_lut, np.array([r["measured"]["LUT"] for r in rows_]), rcond=None)[0]
            c_ff = np.linalg.lstsq(x_ff, np.array([r["measured"]["FF"] for r in rows_]), rcond=None)[0]
            return c_lut, c_ff

        def predict(c_lut, c_ff, row):
            x_lut, x_ff = cls._design([row])
            return {"DSP": row["DSP_pred"], "LUT": float(x_lut[0] @ c_lut), "FF": float(x_ff[0] @ c_ff)}

        def rel(pred, meas):
            return {k: (pred[k] - meas[k]) / meas[k] if meas[k] else (0.0 if pred[k] == 0 else float("inf"))
                    for k in pred}

        c_lut, c_ff = solve(fit)
        model = cls(float(c_lut[0]), float(c_lut[1]), float(c_ff[0]), float(c_ff[1]), float(c_ff[2]),
                    dsp_min_bits=dsp_min_bits, dsp_min_csd=dsp_min_csd)
        enough = len(fit) > 3  # leave-one-out needs more points than FF coefficients
        for row in rows:
            pred = predict(c_lut, c_ff, row)
            entry = {"name": row["name"], "in_domain": row["in_domain"], "predicted": pred,
                     "measured": row["measured"], "rel_err": rel(pred, row["measured"]), "loo_rel_err": None}
            if row["in_domain"] and enough:
                cl, cf = solve([q for q in fit if q["name"] != row["name"]])
                entry["loo_rel_err"] = rel(predict(cl, cf, row), row["measured"])
            model.calibration.append(entry)
        return model

    def calibration_table(self) -> List[Dict[str, Any]]:
        """Measured vs predicted per synthesis point (fit and leave-one-out relative errors)."""
        out = []
        for c in self.calibration:
            row = {"point": c["name"], "in_domain": c["in_domain"]}
            for k in ("DSP", "LUT", "FF"):
                row[f"{k}_measured"] = c["measured"][k]
                row[f"{k}_predicted"] = c["predicted"][k]
                row[f"{k}_rel_err"] = c["rel_err"][k]
                row[f"{k}_loo_rel_err"] = c["loo_rel_err"][k] if c["loo_rel_err"] else float("nan")
            out.append(row)
        return out


@lru_cache(maxsize=1)
def get_cost_model() -> CostModel:
    return CostModel.from_calibration(CALIBRATION_PATH)


def _followed_by_relu(model: tf.keras.Model, layer) -> bool:
    """True if the layer output goes through a ReLU (fused FP32 activation or a following QActivation)."""
    if getattr(getattr(layer, "activation", None), "__name__", "") == "relu":
        return True
    layers = list(model.layers)
    i = layers.index(layer)
    if i + 1 < len(layers) and layers[i + 1].__class__.__name__ == "QActivation":
        return "relu" in str(getattr(layers[i + 1], "quantizer", layers[i + 1].activation))
    return False


def layer_specs_from_model(
    model: tf.keras.Model,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Union[int, Dict[str, int]]] = 8,
    input_bits: int = 8,
    default_weight_bits: int = 8,
    default_act_bits: int = 8,
) -> List[LayerSpec]:
    """LayerSpecs of a (FP32 or QKeras) model, quantizing its float weights with the model's quantizer.

    For an FP32 model this is post-training quantization of the reference weights, which is how the
    cost of a configuration is predicted before it is trained.
    """
    weights_bits = weights_bits or {}
    if isinstance(act_bits, dict):
        act_map, fill = dict(act_bits), default_act_bits
    else:
        act_map, fill = {}, default_act_bits if act_bits is None else int(act_bits)
    layers = [model.get_layer(name) for name in _compute_layer_names(model)]
    a_in = layer_input_bits([l.name for l in layers], act_map, input_bits, fill)
    specs = []
    for layer in layers:
        w_bits = int(weights_bits.get(layer.name, default_weight_bits))
        kernel = layer.kernel.numpy()
        n_out = kernel.shape[-1]
        bias = layer.bias.numpy() if getattr(layer, "bias", None) is not None else np.zeros(n_out, np.float32)
        positions = int(layer.output_shape[1]) if "Conv1D" in layer.__class__.__name__ else 1
        specs.append(LayerSpec(
            layer.name,
            quantize_weights_int(kernel, w_bits).reshape(-1, n_out),
            quantize_weights_int(bias, w_bits),
            w_bits,
            a_in[layer.name],
            positions,
            relu_after=_followed_by_relu(model, layer),
        ))
    return specs


def calibrated_resources(
    model: tf.keras.Model,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Union[int, Dict[str, int]]] = 8,
    input_bits: int = 8,
    cost_model: Optional[CostModel] = None,
) -> Dict[str, Any]:
    """Calibrated VU13P resources of a model: totals {DSP, LUT, FF, BRAM_bits, RAMB18} and per layer."""
    cm = cost_model or get_cost_model()
    specs = layer_specs_from_model(model, weights_bits, act_bits, input_bits)
    total = cm.estimate(specs)
    return {
        "total": {k: float(total[k]) for k in ("DSP", "LUT", "FF", "BRAM_bits", "RAMB18")},
        "layers": {s.name: cm.layer_resources(s) for s in specs},
    }


def make_cost_fn(
    model: tf.keras.Model,
    input_bits: int = 8,
    cost_model: Optional[CostModel] = None,
) -> Callable[[Dict[str, int], Dict[str, int]], Dict[str, float]]:
    """Cost function (weights_bits, act_bits) -> {DSP, LUT, FF, BRAM_bits} for the bit allocator.

    Uses the float weights of ``model`` (e.g. the FP32 reference) quantized at each candidate
    precision. Each layer depends only on (w_l, a_in,l), so per-layer results are cached.
    """
    cm = cost_model or get_cost_model()
    layers = [model.get_layer(name) for name in _compute_layer_names(model)]
    names = [l.name for l in layers]
    cache: Dict[Tuple[str, int, int], Dict[str, float]] = {}

    def layer_cost(layer, w_bits: int, a_in_bits: int) -> Dict[str, float]:
        key = (layer.name, w_bits, a_in_bits)
        if key not in cache:
            spec = layer_specs_from_model(model, {layer.name: w_bits}, {}, input_bits)[names.index(layer.name)]
            spec.a_in_bits = a_in_bits
            cache[key] = cm.layer_resources(spec)
        return cache[key]

    def cost_fn(weights_bits: Dict[str, int], act_bits: Dict[str, int]) -> Dict[str, float]:
        a_in = layer_input_bits(names, act_bits, input_bits)
        out = {"DSP": 0.0, "LUT": 0.0, "FF": cm.gamma_ff, "BRAM_bits": cm.ramb18 * RAMB18_BITS}
        for layer in layers:
            res = layer_cost(layer, int(weights_bits[layer.name]), a_in[layer.name])
            for key in ("DSP", "LUT", "FF"):
                out[key] += res[key]
        return out

    return cost_fn


def _format_ap_fixed(q_info: Optional[Dict[str, Any]], default_bits: int = 8, default_int: int = 2) -> str:
    """Formats a QTools quantizer dictionary as an HLS C++ ap_fixed / ap_ufixed type string."""
    if not q_info or not isinstance(q_info, dict):
        return f"ap_fixed<{default_bits}, {default_int}>"
    bits = int(q_info.get("bits", default_bits))
    int_bits = int(q_info.get("int_bits", default_int))
    is_signed = bool(q_info.get("is_signed", True))
    prefix = "ap_fixed" if is_signed else "ap_ufixed"
    return f"{prefix}<{bits}, {int_bits}>"


def extract_qtools_hardware_profile(
    model: tf.keras.Model,
    input_bits: int = 8,
    input_int_bits: int = 0,
    process: str = "horowitz",
    weights_on_memory: str = "sram",
    activations_on_memory: str = "sram",
) -> Dict[str, Any]:
    """Runs official qkeras.qtools graph analysis to infer exact multiplier/accumulator bitwidths and energy.

    Args:
        model: Compiled Keras or QKeras Model.
        input_bits: Bitwidth of the normalized input calorimeter rings.
        input_int_bits: Integer bitwidth of the input quantizer.
        process: CMOS energy model process ('horowitz' 45nm reference).
        weights_on_memory: Memory placement mode for weights ('sram', 'dram', 'fixed').
        activations_on_memory: Memory placement mode for activations ('sram', 'dram', 'fixed').

    Returns:
        Dictionary with per-layer QTools data types (input, weight, multiplier, accumulator)
        and energy consumption in picojoules (pJ).
    """
    has_qkeras_layers = any(
        layer.__class__.__name__.startswith("Q") for layer in model.layers
    )
    for_reference = not has_qkeras_layers

    source_q = [qkeras.quantized_bits(bits=input_bits, integer=input_int_bits, keep_negative=False, alpha=1)]
    q = run_qtools.QTools(
        model,
        process=process,
        source_quantizers=source_q,
        is_inference=True,
        weights_path=None,
        keras_quantizer=None,
        keras_accumulator=None,
        for_reference=for_reference,
    )

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        q.qtools_stats_to_json(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            stats_json = json.load(f)

    energy_dict = q.pe(
        weights_on_memory=weights_on_memory,
        activations_on_memory=activations_on_memory,
        min_sram_size=0,
        rd_wr_on_io=False,
    )
    total_energy_no_io = float(
        q.extract_energy_sum(qtools_settings.cfg.include_energy, energy_dict)
    )
    total_energy_with_io = float(energy_dict.get("total_cost", total_energy_no_io))

    layers_profile: Dict[str, Dict[str, Any]] = {}
    for layer_name, l_stats in stats_json.items():
        if layer_name == "source_quantizers" or not isinstance(l_stats, dict):
            continue

        in_q_list = l_stats.get("input_quantizer_list", [])
        in_q = in_q_list[0] if in_q_list else None
        w_q = l_stats.get("weight_quantizer")
        b_q = l_stats.get("bias_quantizer")
        mul_q = l_stats.get("multiplier")
        acc_q = l_stats.get("accumulator")
        out_q = l_stats.get("output_quantizer")

        l_energy_info = energy_dict.get(layer_name, {}).get("energy", {})
        e_in = float(l_energy_info.get("inputs", 0.0))
        e_out = float(l_energy_info.get("outputs", 0.0))
        e_param = float(l_energy_info.get("parameters", 0.0))
        e_op = float(l_energy_info.get("op_cost", 0.0))
        e_total = round(e_in + e_out + e_param + e_op, 2)

        layers_profile[layer_name] = {
            "layer_type": l_stats.get("layer_type", ""),
            "operation_count": int(l_stats.get("operation_count", 0)),
            "input_bits": int(in_q.get("bits", input_bits)) if in_q else input_bits,
            "input_int_bits": int(in_q.get("int_bits", input_int_bits)) if in_q else input_int_bits,
            "weight_bits": int(w_q.get("bits", 32 if for_reference else 8)) if w_q else None,
            "weight_int_bits": int(w_q.get("int_bits", 0)) if w_q else None,
            "bias_bits": int(b_q.get("bits", 32 if for_reference else 8)) if b_q else None,
            "mult_bits": int(mul_q.get("bits", 32)) if mul_q else None,
            "mult_int_bits": int(mul_q.get("int_bits", 0)) if mul_q else None,
            "acc_bits": int(acc_q.get("bits", 32)) if acc_q else None,
            "acc_int_bits": int(acc_q.get("int_bits", 0)) if acc_q else None,
            "output_bits": int(out_q.get("bits", 8)) if out_q else None,
            "ap_fixed_input": _format_ap_fixed(in_q, input_bits, input_int_bits),
            "ap_fixed_weight": _format_ap_fixed(w_q) if w_q else "-",
            "ap_fixed_mult": _format_ap_fixed(mul_q) if mul_q else "-",
            "ap_fixed_acc": _format_ap_fixed(acc_q) if acc_q else "-",
            "ap_fixed_output": _format_ap_fixed(out_q) if out_q else "-",
            "energy_pj": {
                "inputs_pj": e_in,
                "outputs_pj": e_out,
                "parameters_pj": e_param,
                "mac_op_pj": e_op,
                "total_layer_pj": e_total,
            },
        }

    return {
        "process": process,
        "for_reference_fp32": for_reference,
        "total_energy_pj": total_energy_no_io,
        "total_energy_with_io_pj": total_energy_with_io,
        "layers": layers_profile,
    }


def estimate_hardware_resources(
    model_config: List[Dict[str, Any]],
) -> Dict[str, float]:
    """MAC-based analytical VU13P estimate (DSP, LUT, FF, BRAM_bits) from a layer config.

    It does not see weight values, adder trees or the hls4ml mapping and under-predicts synthesized
    LUTs by 3x-24x; use calibrated_resources / make_cost_fn for decisions.

    Follows UltraScale+ synthesis rules:
    - If w_bits <= 8 and a_bits <= 8: sub-byte arithmetic is mapped onto CLB LUTs:
      LUT_l = macc_ops * (w_bits * a_bits) / 18.0
    - If w_bits > 8 or a_bits > 8: arithmetic uses DSP48E2 slices:
      DSP_l = macc_ops / 64.0 (pipelined initiation interval factor) + control LUTs
    - Flip-Flops (FF) and BRAM storage bits scale with parameter storage (params * w_bits)
      and datapath pipeline registers.

    Args:
        model_config: List of dicts with keys {'w_bits', 'a_bits', 'params', 'macc'}, where
            'a_bits' is the precision of the activation ENTERING the layer (see layer_input_bits).

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


def estimate_vu13p_detailed_resources(
    model: tf.keras.Model,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Union[int, Dict[str, int]]] = 8,
    reuse_factor: Union[int, Dict[str, int]] = 1,
    clock_mhz: float = 250.0,
    input_dim: Optional[int] = None,
    input_bits: int = 8,
) -> Dict[str, Any]:
    """High-fidelity VU13P resource, accumulator, energy, and latency estimator powered by qkeras.qtools.

    Combines QTools exact data-type inference (multiplier & accumulator bitwidths and CMOS/FPGA energy)
    with HLS microarchitectural modeling (Reuse Factor RF_l, adder reduction trees, 1D streaming line
    buffers, LUTRAM vs. BRAM_36K blocks, and cycle-accurate pipeline latency at `clock_mhz`).

    Args:
        model: Built Keras/QKeras Model.
        weights_bits: Optional per-layer weight bitwidth override mapping.
        act_bits: Global activation bitwidth or mapping keyed by the layer whose ReLU output is
            quantized ({"conv1d_1": a_1, "dense_1": a_2}). Each layer is costed with the
            precision of its input (see layer_input_bits).
        reuse_factor: Global HLS ReuseFactor (II) or per-layer dict mapping.
        clock_mhz: Target FPGA clock frequency in MHz on the VU13P (default 250 MHz -> 4.0 ns period).
        input_dim: Optional input ring count (inferred from model if None).
        input_bits: Precision of the rings entering the first layer.

    Returns:
        Dictionary containing detailed per-layer microarchitectural breakdown, QTools profile,
        energy summary, and pipeline latency in clock cycles and nanoseconds.
    """
    if weights_bits is None:
        weights_bits = {}
    if isinstance(act_bits, int):
        act_bits_map: Dict[str, int] = {}
        fallback_act = act_bits
    elif isinstance(act_bits, dict):
        act_bits_map = act_bits
        fallback_act = 8
    else:
        act_bits_map = {}
        fallback_act = 8

    if input_dim is None:
        in_shape = model.input_shape
        input_dim = in_shape[-1] if in_shape is not None else 25

    a_in = layer_input_bits(_compute_layer_names(model), act_bits_map, input_bits, fallback_act)
    qtools_profile = extract_qtools_hardware_profile(model, input_bits=input_bits)
    qtools_layers = qtools_profile["layers"]

    dummy_x = tf.zeros((1, input_dim))
    curr_tensor = dummy_x
    layer_output_shapes = {}
    for layer in model.layers:
        curr_tensor = layer(curr_tensor)
        layer_output_shapes[layer.name] = curr_tensor.shape

    clk_period_ns = 1000.0 / float(clock_mhz)
    detailed_layers: List[Dict[str, Any]] = []

    tot_lut_mult = 0.0
    tot_lut_adder = 0.0
    tot_ff_linebuf = 0.0
    tot_ff_datapath = 0.0
    tot_bram_36k = 0
    tot_lutram_bits = 0
    tot_cycles = 0

    for layer in model.layers:
        layer_cls = layer.__class__.__name__
        if ("Conv1D" in layer_cls or "Dense" in layer_cls) and hasattr(layer, "kernel"):
            lname = layer.name
            out_shape = layer_output_shapes[lname]
            kernel = layer.kernel
            q_layer = qtools_layers.get(lname, {})

            if "Conv1D" in layer_cls:
                l_out = int(out_shape[1])
                k_size = int(kernel.shape[0])
                c_in = int(kernel.shape[1])
                c_out = int(kernel.shape[2])
                macs = l_out * k_size * c_in * c_out
                fan_in = k_size * c_in
                n_out_elems = l_out * c_out
                line_buffer_ffs = float(k_size * c_in * a_in[lname] * c_out)
            else:
                n_in = int(kernel.shape[0])
                n_out = int(kernel.shape[1])
                macs = n_in * n_out
                fan_in = n_in
                n_out_elems = n_out
                line_buffer_ffs = 0.0

            rf_l = (
                int(reuse_factor.get(lname, 1))
                if isinstance(reuse_factor, dict)
                else int(reuse_factor)
            )
            rf_l = max(1, rf_l)

            b_w = int(weights_bits.get(lname, q_layer.get("weight_bits") or 8))
            b_a = a_in[lname]
            num_weights = int(layer.count_params())

            # Analytical or QTools-extracted multiplier and accumulator widths
            mult_bits = int(q_layer.get("mult_bits") or (b_w + b_a))
            acc_bits = int(
                q_layer.get("acc_bits")
                or (b_w + b_a + math.ceil(math.log2(max(fan_in, 2))))
            )
            acc_int_bits = int(
                q_layer.get("acc_int_bits")
                or min(acc_bits - 1, 4 + math.ceil(math.log2(max(fan_in, 2))))
            )

            active_mults = math.ceil(float(macs) / float(rf_l))
            adder_tree_depth = math.ceil(math.log2(max(fan_in, 2)))

            # Base VU13P resource estimation (strictly consistent with estimate_hardware_resources)
            base_vu13p = estimate_hardware_resources([{
                "w_bits": b_w,
                "a_bits": b_a,
                "params": num_weights,
                "macc": macs,
            }])

            # Microarchitectural LUT decomposition (Multiplier LUTs vs. Adder-Tree LUTs at RF_l)
            lut_mult_rf = (active_mults * (b_w * b_a) / 18.0) if (b_w <= 8 and b_a <= 8) else (active_mults * (b_w + b_a) / 36.0)
            lut_adder_rf = active_mults * float(acc_bits) * 0.35
            ff_datapath_rf = float(num_weights * b_w) + float(n_out_elems * acc_bits) / float(rf_l)

            weight_bits_total = num_weights * b_w
            # Vivado maps small ROMs (< 2048 bits) to LUTRAM/distributed registers and larger tables to RAMB36E2
            if weight_bits_total >= 2048 and rf_l > 1:
                bram_36k_blocks = math.ceil(weight_bits_total / (36.0 * 1024.0))
                lutram_bits = 0
            else:
                bram_36k_blocks = 0
                lutram_bits = weight_bits_total

            latency_cycles = rf_l + adder_tree_depth + 1
            latency_ns = round(latency_cycles * clk_period_ns, 2)
            layer_energy_pj = float(
                q_layer.get("energy_pj", {}).get("total_layer_pj", 0.0)
            )

            tot_lut_mult += lut_mult_rf
            tot_lut_adder += lut_adder_rf
            tot_ff_linebuf += line_buffer_ffs
            tot_ff_datapath += ff_datapath_rf
            tot_bram_36k += bram_36k_blocks
            tot_lutram_bits += lutram_bits
            tot_cycles += latency_cycles

            detailed_layers.append({
                "layer_name": lname,
                "layer_type": layer_cls,
                "params": num_weights,
                "macs": macs,
                "fan_in": fan_in,
                "reuse_factor": rf_l,
                "active_multipliers": active_mults,
                "b_w": b_w,
                "b_a": b_a,
                "mult_bits": mult_bits,
                "acc_bits": acc_bits,
                "acc_int_bits": acc_int_bits,
                "ap_fixed_weight": q_layer.get("ap_fixed_weight", f"ap_fixed<{b_w}, 2>"),
                "ap_fixed_mult": q_layer.get("ap_fixed_mult", f"ap_fixed<{mult_bits}, 4>"),
                "ap_fixed_acc": q_layer.get("ap_fixed_acc", f"ap_fixed<{acc_bits}, {acc_int_bits}>"),
                "DSP": base_vu13p["DSP"],
                "LUT": base_vu13p["LUT"],
                "LUT_mult_rf": round(lut_mult_rf, 1),
                "LUT_adder_tree_rf": round(lut_adder_rf, 1),
                "FF": base_vu13p["FF"],
                "FF_line_buffer": round(line_buffer_ffs, 1),
                "FF_datapath_rf": round(ff_datapath_rf, 1),
                "BRAM_bits": base_vu13p["BRAM_bits"],
                "LUTRAM_bits": lutram_bits,
                "BRAM_36K_blocks": bram_36k_blocks,
                "adder_tree_depth": adder_tree_depth,
                "latency_cycles": latency_cycles,
                "latency_ns": latency_ns,
                "energy_pj": layer_energy_pj,
            })

    return {
        "clock_mhz": clock_mhz,
        "clock_period_ns": round(clk_period_ns, 3),
        "total_latency_cycles": tot_cycles,
        "total_latency_ns": round(tot_cycles * clk_period_ns, 2),
        "total_energy_pj": qtools_profile["total_energy_pj"],
        "total_energy_with_io_pj": qtools_profile["total_energy_with_io_pj"],
        "microarch_totals_at_rf": {
            "LUT_mult": round(tot_lut_mult, 1),
            "LUT_adder_tree": round(tot_lut_adder, 1),
            "FF_line_buffer": round(tot_ff_linebuf, 1),
            "FF_datapath": round(tot_ff_datapath, 1),
            "LUTRAM_bits": tot_lutram_bits,
            "BRAM_36K_blocks": tot_bram_36k,
        },
        "layers": detailed_layers,
        "qtools_profile": qtools_profile,
    }


def compute_model_hardware_costs(
    model: tf.keras.Model,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Union[int, Dict[str, int]]] = 8,
    default_act_bits: int = 8,
    default_weight_bits: int = 8,
    input_dim: Optional[int] = None,
    reuse_factor: Union[int, Dict[str, int]] = 1,
    clock_mhz: float = 250.0,
    input_bits: int = 8,
) -> Dict[str, Any]:
    """Compute theoretical hardware costs, Virtex UltraScale+ VU13P resources, and QTools metrics.

    Maintains 100% backward compatibility with existing keys ('total_macs', 'total_bops',
    'total_weight_bits', 'total_params', 'vu13p_resources', 'breakdown') while enriching each
    layer and the top-level dictionary with `qkeras.qtools` accumulator bitwidths, energy (pJ),
    and VU13P pipeline latency (cycles and ns).

    ``act_bits`` is keyed by the layer whose ReLU output is quantized; each layer is costed with
    the precision of its input (layer_input_bits), reported as ``b_a`` in the breakdown, so
    BOPs = sum_l MAC_l * w_l * a_in,l. ``input_bits`` is the precision of the input rings.

    ``vu13p_resources`` and the per-layer DSP/LUT/FF come from the calibrated structural model
    (calibrated_resources) applied to the model's weights quantized at ``weights_bits``.
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

    if input_dim is None:
        in_shape = model.input_shape
        input_dim = in_shape[-1] if in_shape is not None else 50

    a_in = layer_input_bits(_compute_layer_names(model), act_bits_map, input_bits, fallback_act)
    detailed_qtools = estimate_vu13p_detailed_resources(
        model=model,
        weights_bits=weights_bits,
        act_bits=act_bits,
        reuse_factor=reuse_factor,
        clock_mhz=clock_mhz,
        input_dim=input_dim,
        input_bits=input_bits,
    )
    detailed_by_name = {item["layer_name"]: item for item in detailed_qtools["layers"]}

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
    calibrated = calibrated_resources(model, weights_bits, act_bits, input_bits)

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
            b_a = a_in[layer.name]
            bops = macs * b_w * b_a
            num_weights = int(layer.count_params())
            weight_bits = num_weights * b_w

            layer_vu13p = calibrated["layers"][layer.name]

            q_det = detailed_by_name.get(layer.name, {})

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
                "mult_bits": q_det.get("mult_bits", b_w + b_a),
                "acc_bits": q_det.get("acc_bits", b_w + b_a + 4),
                "acc_int_bits": q_det.get("acc_int_bits", 4),
                "ap_fixed_weight": q_det.get("ap_fixed_weight", f"ap_fixed<{b_w}, 2>"),
                "ap_fixed_acc": q_det.get("ap_fixed_acc", f"ap_fixed<{b_w + b_a + 4}, 4>"),
                "bops": bops,
                "params": num_weights,
                "DSP": layer_vu13p["DSP"],
                "LUT": layer_vu13p["LUT"],
                "FF": layer_vu13p["FF"],
                "BRAM_bits": 0.0,  # weights are constants in the logic; the sigmoid RAMB18 is in the total
                "adder_bits": layer_vu13p["adder_bits"],
                "nonzero_weight_frac": layer_vu13p["nonzero_frac"],
                "latency_cycles": q_det.get("latency_cycles", 2),
                "latency_ns": q_det.get("latency_ns", 8.0),
                "energy_pj": q_det.get("energy_pj", 0.0),
            })

    vu13p_resources = {k: calibrated["total"][k] for k in ("DSP", "LUT", "FF", "BRAM_bits")}

    return {
        "total_macs": total_macs,
        "total_bops": total_bops,
        "total_weight_bits": total_weight_bits,
        "total_params": int(model.count_params()),
        "total_energy_pj": detailed_qtools["total_energy_pj"],
        "total_energy_with_io_pj": detailed_qtools["total_energy_with_io_pj"],
        "total_latency_cycles": detailed_qtools["total_latency_cycles"],
        "total_latency_ns": detailed_qtools["total_latency_ns"],
        "clock_mhz": clock_mhz,
        "vu13p_resources": vu13p_resources,
        "qtools_vu13p_detailed": detailed_qtools,
        "breakdown": layer_breakdown,
    }
