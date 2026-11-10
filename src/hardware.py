"""Hardware cost model of the compact CNN1D on the AMD Virtex UltraScale+ VU13P (hls4ml io_parallel).

Why this model (docs/theory_and_formulation.md, Sec. 4)
-------------------------------------------------------
The previous rules (LUT = MAC*w*a/18, DSP = ceil(MAC/64), BRAM = params*w) were checked against
hls4ml -> Vitis HLS -> Vivado place & route of this very network (repository cnn-fpga-research) and
missed by 3x-54x: in an ``io_parallel`` / ``Latency`` / ``ReuseFactor=1`` design the weights are
constants wired into the logic, so

* the dominant cost is the adder tree of every neuron, not a generic w x a multiplier;
* a zero weight costs nothing and a power-of-two weight is a shift (wiring);
* a constant multiplier with d nonzero canonical-signed-digit (CSD) digits is (d - 1) adders,
  unless it is wide (> ~10 bits) and non-trivial, in which case HLS uses one DSP48E2;
* weights are never stored in BRAM; the only RAM block is the sigmoid lookup table.

Structural model, per layer l (positions P_l computed in parallel, fan-in n_l, output neurons o):

    adders_l     = P_l * sum_o [ (terms_o - 1) + sum_{k in o, LUT-mult} (csd(k) - 1) ]
    width_l      = a_in,l + w_l + ceil(log2 n_l) / 2          (mean width along the tree)
    adder_bits_l = adders_l * width_l
    relu_bits_l  = P_l * n_out,l * (a_in,l + w_l + ceil(log2 n_l))   if a ReLU follows l, else 0
    DSP_l        = P_l * #{k : max(w_l, a_in,l) > DSP_MIN_BITS and csd(k) >= DSP_MIN_CSD}

    LUT      = beta_LUT * sum_l adder_bits_l + gamma_LUT * sum_l relu_bits_l
    FF       = beta_FF  * sum_l adder_bits_l + delta_FF * DSP + gamma_FF
    BRAM_18K = 1 (sigmoid table)

relu_bits counts the accumulator bits that enter the ReLU (+ max-pooling) of a layer: in the small
designs this "glue" (~3-5 k LUT) does not shrink with the adder trees, and without it the model
under-predicted the out-of-sample points by up to 35 %. Coefficients are fitted by least squares on
the synthesized points stored in ``src/calibration/vu13p_hls4ml_synthesis.json`` and the expected
accuracy is the leave-one-out error (``calibration_table``); ``margins`` gives the worst LOO
under-prediction per resource, to be used as a safety margin on the budgets. DSP_MIN_BITS = 10 follows
the hls4ml literature (multiplications below ~10 bits go to LUTs); DSP_MIN_CSD = 3 (constants with <= 2
CSD digits are a single shift-add) reproduces the 16-bit point (1,245 predicted vs 1,241 DSPs).
Validity domain: hls4ml 1.3 io_parallel, Latency, ReuseFactor = 1, Conv1D ParallelizationFactor =
output width (fully parallel, II = 1), 240 MHz, VU13P, this topology. Outside it (e.g. PF = 1) the
model is not calibrated: add synthesis points.

Activation precision convention
-------------------------------
The multiplications of layer l use the precision of the tensor that *enters* l: the network
input for ``conv1d_1``, the ReLU output of ``conv1d_1`` for ``dense_1`` and the ReLU output of
``dense_1`` for ``output_dense``. ``act_bits`` dicts are keyed by the layer whose output is
quantized (``{"conv1d_1": a_1, "dense_1": a_2}``), so ``a_in`` of a layer is ``act_bits`` of the
*previous* compute layer. (The previous implementation used ``act_bits[layer]`` for the layer
itself and fell back to 8 bits for ``output_dense``.)
"""

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.models import quantize_weights_int

# AMD/Xilinx Virtex UltraScale+ VU13P (DS923, Table "Virtex UltraScale+ FPGA Feature Summary")
VU13P_CHIP_LIMITS: Dict[str, float] = {
    "DSP": 12288.0,       # DSP48E2 slices
    "LUT": 1728000.0,     # CLB LUTs
    "FF": 3456000.0,      # CLB flip-flops
    "BRAM_18K": 5376.0,   # 2,688 x RAMB36 = 5,376 x RAMB18 (94.5 Mb)
    "URAM": 1280.0,       # 288 Kb blocks (360 Mb)
}
# Resources that enter the optimization as constraints g_j(x) <= 0. BRAM_18K is constant
# (1 sigmoid table) in io_parallel and is reported only.
CONSTRAINED_RESOURCES = ("DSP", "LUT", "FF")
VU13P_LIMITS: Dict[str, float] = {k: VU13P_CHIP_LIMITS[k] for k in CONSTRAINED_RESOURCES}

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration" / "vu13p_hls4ml_synthesis.json"
COMPUTE_LAYER_CLASSES = ("Conv1D", "Dense")


def get_vu13p_budget(
    n_instances: int = 1,
    device_fraction: float = 1.0,
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Per-instance budget B_j = device_fraction * chip_j / n_instances for DSP, LUT and FF.

    ``n_instances`` copies of the classifier share ``device_fraction`` of the VU13P (the rest is
    left to the other trigger firmware). ``overrides`` sets explicit absolute limits.
    """
    budgets = {k: float(device_fraction * VU13P_CHIP_LIMITS[k] / n_instances) for k in CONSTRAINED_RESOURCES}
    budgets.update({k: float(v) for k, v in (overrides or {}).items()})
    return budgets


def csd_digits(n: int) -> int:
    """Number of nonzero digits of |n| in canonical signed-digit form (0 for n = 0, 1 for powers of two)."""
    n = abs(int(n))
    count = 0
    while n:
        if n & 1:
            n -= 2 - (n & 3)  # digit +1 or -1
            count += 1
        n >>= 1
    return count


@dataclass
class LayerSpec:
    """Everything the cost model needs about one compute layer."""

    name: str
    kernel_int: np.ndarray  # integer weight codes, shape (fan_in, n_out)
    bias_int: np.ndarray    # integer bias codes, shape (n_out,)
    w_bits: int
    a_in_bits: int
    positions: int = 1      # output positions computed in parallel (Conv1D: output width)
    relu_after: bool = False  # a ReLU (+ pooling) consumes this layer's accumulators

    @property
    def fan_in(self) -> int:
        return int(self.kernel_int.shape[0])

    @property
    def macs(self) -> int:
        return int(self.kernel_int.size * self.positions)


def layer_features(spec: LayerSpec, dsp_min_bits: int = 10, dsp_min_csd: int = 3) -> Dict[str, float]:
    """Structural features (adders, adder bits, DSPs) of one layer; see module docstring."""
    k = np.asarray(spec.kernel_int)
    b = np.asarray(spec.bias_int).ravel()
    wide = max(spec.w_bits, spec.a_in_bits) > dsp_min_bits
    csd = np.vectorize(csd_digits, otypes=[np.int64])(k) if k.size else k
    nonzero = k != 0
    dsp_mask = nonzero & wide & (csd >= dsp_min_csd)
    lut_mult_extra = np.where(nonzero & ~dsp_mask, csd - 1, 0)
    terms = nonzero.sum(axis=0) + (b != 0).astype(np.int64)
    adders_per_pos = float(np.maximum(terms - 1, 0).sum() + lut_mult_extra.sum())
    width = spec.a_in_bits + spec.w_bits + math.ceil(math.log2(max(spec.fan_in, 1))) / 2.0
    adders = adders_per_pos * spec.positions
    acc_width = spec.a_in_bits + spec.w_bits + math.ceil(math.log2(max(spec.fan_in, 1)))
    relu_bits = float(spec.positions * k.shape[1] * acc_width) if spec.relu_after else 0.0
    return {
        "adders": adders,
        "width": width,
        "adder_bits": adders * width,
        "relu_bits": relu_bits,
        "DSP": float(dsp_mask.sum() * spec.positions),
        "nonzero_frac": float(nonzero.mean()) if k.size else 0.0,
        "macs": float(spec.macs),
    }


@dataclass
class CostModel:
    """Calibrated VU13P cost model (see module docstring)."""

    beta_lut: float
    gamma_lut: float      # LUT per ReLU-input bit
    beta_ff: float
    delta_ff: float
    gamma_ff: float       # constant FF overhead
    bram18_fixed: float = 1.0
    dsp_min_bits: int = 10
    dsp_min_csd: int = 3
    calibration: List[Dict[str, Any]] = field(default_factory=list)

    def layer_resources(self, spec: LayerSpec) -> Dict[str, float]:
        """Layer-attributable resources (the constant overhead is in ``overhead``)."""
        f = layer_features(spec, self.dsp_min_bits, self.dsp_min_csd)
        return {
            "DSP": f["DSP"],
            "LUT": self.beta_lut * f["adder_bits"] + self.gamma_lut * f["relu_bits"],
            "FF": self.beta_ff * f["adder_bits"] + self.delta_ff * f["DSP"],
            "adder_bits": f["adder_bits"],
            "relu_bits": f["relu_bits"],
            "nonzero_frac": f["nonzero_frac"],
        }

    @property
    def overhead(self) -> Dict[str, float]:
        """Configuration-independent part (I/O, control, sigmoid table)."""
        return {"DSP": 0.0, "LUT": 0.0, "FF": self.gamma_ff, "BRAM_18K": self.bram18_fixed}

    def estimate(self, specs: Sequence[LayerSpec]) -> Dict[str, float]:
        total = dict(self.overhead)
        total["adder_bits"] = total["relu_bits"] = 0.0
        for s in specs:
            r = self.layer_resources(s)
            for key in ("DSP", "LUT", "FF", "adder_bits", "relu_bits"):
                total[key] += r[key]
        return total

    @property
    def margins(self) -> Dict[str, float]:
        """Worst leave-one-out under-prediction per resource (fraction), to inflate predicted costs."""
        out = {}
        for k in ("DSP", "LUT", "FF"):
            errs = [c["loo_rel_err"][k] for c in self.calibration if c["in_domain"] and c.get("loo_rel_err")]
            out[k] = float(max([0.0] + [-e for e in errs if np.isfinite(e)]))
        return out

    @staticmethod
    def _design_matrices(rows):
        ab = np.array([r["adder_bits"] for r in rows])
        rb = np.array([r["relu_bits"] for r in rows])
        dsp = np.array([r["DSP_pred"] for r in rows])
        return np.c_[ab, rb], np.c_[ab, dsp, np.ones_like(ab)]

    @classmethod
    def from_calibration(cls, path: Union[str, Path] = CALIBRATION_PATH, dsp_min_bits: int = 10,
                         dsp_min_csd: int = 3) -> "CostModel":
        """Least-squares fit on the fully parallel (PF = output width) points, with leave-one-out errors."""
        points = json.loads(Path(path).read_text())["points"]
        rows = []
        for p in points:
            specs = [
                LayerSpec(name, np.asarray(l["kernel_int"]), np.asarray(l["bias_int"]), l["w_bits"], l["a_in_bits"],
                          l["positions_parallel"], l.get("relu_after", name in ("conv1d_1", "dense_1")))
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
        x_lut, x_ff = cls._design_matrices(fit)
        lut = np.array([r["measured"]["LUT"] for r in fit])
        ff = np.array([r["measured"]["FF"] for r in fit])
        c_lut = np.linalg.lstsq(x_lut, lut, rcond=None)[0]
        c_ff = np.linalg.lstsq(x_ff, ff, rcond=None)[0]
        model = cls(float(c_lut[0]), float(c_lut[1]), float(c_ff[0]), float(c_ff[1]), float(c_ff[2]),
                    dsp_min_bits=dsp_min_bits, dsp_min_csd=dsp_min_csd)

        def predict(coef_lut, coef_ff, r):
            xl, xf = cls._design_matrices([r])
            return {"DSP": r["DSP_pred"], "LUT": float(xl[0] @ coef_lut), "FF": float(xf[0] @ coef_ff)}

        def rel(pred, meas):
            return {k: (pred[k] - meas[k]) / meas[k] if meas[k] else (0.0 if pred[k] == 0 else float("inf"))
                    for k in pred}

        enough = len(fit) > x_ff.shape[1]  # LOO needs more points than coefficients
        for r in rows:
            pred = predict(c_lut, c_ff, r)
            entry = {"name": r["name"], "in_domain": r["in_domain"], "predicted": pred, "measured": r["measured"],
                     "rel_err": rel(pred, r["measured"]), "loo_rel_err": None}
            if r["in_domain"] and enough:
                rest = [q for q in fit if q["name"] != r["name"]]
                xl, xf = cls._design_matrices(rest)
                cl = np.linalg.lstsq(xl, np.array([q["measured"]["LUT"] for q in rest]), rcond=None)[0]
                cf = np.linalg.lstsq(xf, np.array([q["measured"]["FF"] for q in rest]), rcond=None)[0]
                entry["loo_rel_err"] = rel(predict(cl, cf, r), r["measured"])
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
    return CostModel.from_calibration()


def _compute_layers(model) -> List[Any]:
    return [
        l for l in model.layers
        if any(c in l.__class__.__name__ for c in COMPUTE_LAYER_CLASSES) and getattr(l, "kernel", None) is not None
    ]


def _followed_by_relu(model, layer) -> bool:
    """True if the layer's output goes through a ReLU (fused FP32 activation or a following QActivation)."""
    act = getattr(layer, "activation", None)
    if getattr(act, "__name__", "") == "relu":
        return True
    layers = list(model.layers)
    i = layers.index(layer)
    if i + 1 < len(layers) and layers[i + 1].__class__.__name__ == "QActivation":
        return "relu" in str(getattr(layers[i + 1], "quantizer", layers[i + 1].activation))
    return False


def layer_input_bits(
    layer_names: Sequence[str], act_bits: Dict[str, int], input_bits: int, default_act_bits: int = 8
) -> Dict[str, int]:
    """Precision of the tensor entering each compute layer (see 'Activation precision convention')."""
    out, current = {}, int(input_bits)
    for name in layer_names:
        out[name] = current
        current = int(act_bits.get(name, default_act_bits))
    return out


def layer_specs_from_model(
    model,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Union[int, Dict[str, int]]] = None,
    input_bits: int = 8,
    default_weight_bits: int = 8,
    default_act_bits: int = 8,
) -> List[LayerSpec]:
    """LayerSpecs of a (FP32 or QKeras) model, quantizing its float weights with the project quantizer.

    For an FP32 model this is post-training quantization of the reference weights, which predicts the
    synthesized cost of the QAT model within a few percent (zeros/CSD digits barely move during QAT).
    """
    weights_bits = weights_bits or {}
    if act_bits is None or isinstance(act_bits, int):
        fill = default_act_bits if act_bits is None else int(act_bits)
        act_map: Dict[str, int] = {}
    else:
        fill, act_map = default_act_bits, dict(act_bits)
    layers = _compute_layers(model)
    names = [l.name for l in layers]
    a_in = layer_input_bits(names, {n: act_map.get(n, fill) for n in names}, input_bits, fill)
    specs = []
    for layer in layers:
        w_bits = int(weights_bits.get(layer.name, default_weight_bits))
        kernel = layer.kernel.numpy()
        n_out = kernel.shape[-1]
        k_int = quantize_weights_int(kernel, w_bits).reshape(-1, n_out)
        bias = layer.bias.numpy() if getattr(layer, "bias", None) is not None else np.zeros(n_out, np.float32)
        b_int = quantize_weights_int(bias, w_bits)
        positions = int(layer.output_shape[1]) if "Conv1D" in layer.__class__.__name__ else 1
        specs.append(LayerSpec(layer.name, k_int, b_int, w_bits, a_in[layer.name], positions,
                               relu_after=_followed_by_relu(model, layer)))
    return specs


def compute_model_hardware_costs(
    model,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Union[int, Dict[str, int]]] = 8,
    input_bits: int = 8,
    default_act_bits: int = 8,
    default_weight_bits: int = 8,
    cost_model: Optional[CostModel] = None,
) -> Dict[str, Any]:
    """MACs, BOPs, parameters and calibrated VU13P resources (total + per layer) of a model.

    BOPs = sum_l MAC_l * w_l * a_in,l (input-activation precision, see module docstring).
    """
    cm = cost_model or get_cost_model()
    specs = layer_specs_from_model(model, weights_bits, act_bits, input_bits, default_weight_bits, default_act_bits)
    layers = {l.name: l for l in _compute_layers(model)}
    breakdown = []
    for s in specs:
        r = cm.layer_resources(s)
        breakdown.append({
            "layer_name": s.name,
            "layer_type": layers[s.name].__class__.__name__,
            "units": int(s.kernel_int.shape[1]),
            "macs": s.macs,
            "b_w": s.w_bits,
            "b_a_in": s.a_in_bits,
            "bops": s.macs * s.w_bits * s.a_in_bits,
            "params": int(layers[s.name].count_params()),
            "nonzero_weight_frac": r["nonzero_frac"],
            "DSP": r["DSP"],
            "LUT": r["LUT"],
            "FF": r["FF"],
            "adder_bits": r["adder_bits"],
        })
    totals = cm.estimate(specs)
    return {
        "total_macs": int(sum(b["macs"] for b in breakdown)),
        "total_bops": int(sum(b["bops"] for b in breakdown)),
        "total_weight_bits": int(sum(b["params"] * b["b_w"] for b in breakdown)),
        "total_params": int(model.count_params()),
        "vu13p_resources": {k: float(totals[k]) for k in ("DSP", "LUT", "FF", "BRAM_18K")},
        "breakdown": breakdown,
    }
