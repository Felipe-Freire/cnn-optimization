"""Second-order (Hessian) sensitivity of the loss to weight and activation quantization.

Model (docs/theory_and_formulation.md, Sec. 3): around the trained FP32 point (gradient ~ 0),
quantizing tensor T_l with b bits adds the error delta = Q_b(T_l) - T_l and

    Delta L ~ 1/2 delta^T H delta ~ 1/2 * (Tr(H_ll) / n_l) * ||Q_b(T_l) - T_l||^2  =: Omega_l(b)

(cross-layer blocks and off-diagonal terms average out for independent rounding errors; the
average trace Tr(H_ll)/n_l multiplies the *total* squared error, as in HAWQ-V2/V3 and Campos et
al.). Tr(H_ll) is the full trace of the diagonal block of layer l (sum over its n_l entries), estimated
with Hutchinson's method. ||Q_b(T) - T||^2 is computed with the *actual* project quantizer
(src.models), so clipping, the fixed QKeras range and the coarse 2-bit grid are all accounted for.

Fixes w.r.t. the previous implementation:
* c_l no longer multiplies the full trace by n_l (the trace already sums over the n_l weights,
  so the old c_l = Tr(H_l) n_l r_l^2 / 3 counted n_l twice and favored large layers by ~n_l);
* c_l uses the step of the quantizer that is actually trained (fixed range 2^I, I = min(2, b-1)),
  not an adaptive range [-r_l, r_l] that QKeras (alpha=1) does not implement;
* Hutchinson probes one layer block at a time (z_l^T H_ll z_l), instead of a joint vector whose
  cross-layer terms only cancel in expectation and made small-layer estimates negative.
"""

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from src.models import integer_bits, relu_quantizer, weight_quantizer


def _kernel_layers(model, names: Sequence[str] = None):
    layers = [l for l in model.layers if getattr(l, "kernel", None) is not None]
    return [l for l in layers if names is None or l.name in names]


def compute_hutchinson_hessian_trace(
    model,
    loss_fn,
    x_batch,
    y_batch,
    num_hvp: int = 32,
    seed: int = 42,
) -> Dict[str, float]:
    """Tr(H_ll) of the loss w.r.t. each layer kernel, by block-wise Hutchinson probing.

    Tr(H_ll) ~ (1/M) sum_j z_j^T H_ll z_j, z_j Rademacher, one Hessian-vector product per layer and
    draw (double backprop). Returns the full trace (sum over the kernel entries).
    """
    rng = np.random.default_rng(seed)
    layers = _kernel_layers(model)
    traces = {l.name: 0.0 for l in layers}
    for _ in range(num_hvp):
        for layer in layers:
            var = layer.kernel
            z = tf.constant(rng.choice([-1.0, 1.0], size=var.shape).astype(np.float32))
            with tf.GradientTape() as outer:
                with tf.GradientTape() as inner:
                    loss = loss_fn(y_batch, model(x_batch, training=False))
                g = inner.gradient(loss, var)
                gz = tf.reduce_sum(g * z)
            hz = outer.gradient(gz, var)
            traces[layer.name] += float(tf.reduce_sum(z * hz).numpy()) / num_hvp
    return traces


def _split_forward(model, split_name: str, x) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
    """Activation right after layer ``split_name`` (post-ReLU for fused activations) and the head after it."""
    idx = [l.name for l in model.layers].index(split_name)
    h = x
    for layer in model.layers[: idx + 1]:
        h = layer(h, training=False)

    def head(a):
        out = a
        for layer in model.layers[idx + 1:]:
            out = layer(out, training=False)
        return out

    return h, head


def compute_activation_hessian_traces(
    model,
    loss_fn,
    x_batch,
    y_batch,
    layer_names: Iterable[str],
    num_hvp: int = 32,
    seed: int = 7,
) -> Dict[str, Tuple[float, int]]:
    """Tr(H) of the loss w.r.t. the activation tensor produced by each layer (batch x features).

    Returns {layer: (trace, n_entries)}. The loss is a batch mean, so H is block diagonal over
    samples and the trace sums the per-entry curvatures of the whole batch.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name in layer_names:
        a, head = _split_forward(model, name, x_batch)
        a = tf.constant(a)
        tr = 0.0
        for _ in range(num_hvp):
            z = tf.constant(rng.choice([-1.0, 1.0], size=a.shape).astype(np.float32))
            with tf.GradientTape() as outer:
                outer.watch(a)
                with tf.GradientTape() as inner:
                    inner.watch(a)
                    loss = loss_fn(y_batch, head(a))
                g = inner.gradient(loss, a)
                gz = tf.reduce_sum(g * z)
            hz = outer.gradient(gz, a)
            tr += float(tf.reduce_sum(z * hz).numpy()) / num_hvp
        out[name] = (tr, int(np.prod(a.shape)))
    return out


def weight_quantization_errors(model, bits_options: Sequence[int], layer_names: Sequence[str] = None):
    """{layer: {b: ||Q_b(W) - W||^2}} for the kernel of each layer, with the project weight quantizer."""
    out = {}
    for layer in _kernel_layers(model, layer_names):
        w = layer.kernel.numpy()
        out[layer.name] = {
            int(b): float(np.sum((np.asarray(weight_quantizer(b)(w)) - w) ** 2)) for b in bits_options
        }
    return out


def activation_quantization_errors(model, x_batch, bits_options: Sequence[int], layer_names: Iterable[str]):
    """{layer: {b: ||Q_b(A) - A||^2}} over the batch, with the project ReLU quantizer."""
    out = {}
    for name in layer_names:
        a, _ = _split_forward(model, name, x_batch)
        a = a.numpy()
        out[name] = {int(b): float(np.sum((np.asarray(relu_quantizer(b)(a)) - a) ** 2)) for b in bits_options}
    return out


def omega_table(traces: Dict[str, float], sizes: Dict[str, int], errors: Dict[str, Dict[int, float]]):
    """Omega_l(b) = 1/2 * max(Tr_l, 0) / n_l * ||Q_b - T||^2 for every layer and bit width."""
    return {
        name: {b: 0.5 * max(traces[name], 0.0) / sizes[name] * err for b, err in errs.items()}
        for name, errs in errors.items()
    }


def compute_layer_c_coefficients(
    model,
    traces: Dict[str, float],
    range_mode: str = "qkeras",
) -> Dict[str, float]:
    """c_l of the continuous model Omega_l(b) ~ c_l 4^(-b) (uniform rounding noise, Delta = kappa 2^-b).

    E||Q_b(W)-W||^2 = n_l Delta^2 / 12  and  Omega_l = 1/2 (Tr_l / n_l) E||.||^2 = Tr_l kappa^2 4^-b / 24,
    so c_l = Tr_l * kappa_l^2 / 24 (no n_l factor). kappa_l = 2^(I+1) for the trained QKeras quantizer
    (range_mode="qkeras", I = 2 for b >= 3) or 2 r_l for an adaptive range [-r_l, r_l] ("adaptive").
    """
    c = {}
    for layer in _kernel_layers(model):
        if layer.name not in traces:
            continue
        if range_mode == "qkeras":
            kappa = 2.0 ** (integer_bits(8) + 1)
        elif range_mode == "adaptive":
            kappa = 2.0 * (float(np.max(np.abs(layer.kernel.numpy()))) + 1e-12)
        else:
            raise ValueError(f"unknown range_mode {range_mode!r}")
        c[layer.name] = max(traces[layer.name], 0.0) * kappa ** 2 / 24.0
    return c


def measure_ptq_loss_increase(
    model,
    loss_fn,
    x_eval,
    y_eval,
    bits_options: Sequence[int],
    weight_layers: Sequence[str],
    act_layers: Sequence[str],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Measured Delta L when only one tensor is quantized (post-training), to validate Omega.

    Returns {"weights": {layer: {b: dL}}, "activations": {layer: {b: dL}}}.
    """
    base = float(loss_fn(y_eval, model(x_eval, training=False)).numpy())
    res = {"weights": {}, "activations": {}}
    for layer in _kernel_layers(model, weight_layers):
        orig = layer.get_weights()
        res["weights"][layer.name] = {}
        for b in bits_options:
            layer.set_weights([np.asarray(weight_quantizer(b)(orig[0]))] + orig[1:])
            res["weights"][layer.name][int(b)] = float(loss_fn(y_eval, model(x_eval, training=False)).numpy()) - base
        layer.set_weights(orig)
    for name in act_layers:
        a, head = _split_forward(model, name, x_eval)
        res["activations"][name] = {
            int(b): float(loss_fn(y_eval, head(relu_quantizer(b)(a))).numpy()) - base for b in bits_options
        }
    return res


def monotone_table(table: Dict[str, Dict[int, float]]) -> Dict[str, Dict[int, float]]:
    """Clip at 0 and make each Omega_l(b) non-increasing in b (more bits never cost loss).

    Measured single-tensor loss increases are noisy at high precision (they can be slightly negative);
    taking the running maximum from high to low bits keeps the information and removes the noise.
    """
    out = {}
    for name, tab in table.items():
        running, mono = 0.0, {}
        for b in sorted(tab, reverse=True):
            running = max(running, float(tab[b]), 0.0)
            mono[b] = running
        out[name] = {b: mono[b] for b in sorted(mono)}
    return out


def sensitivity_report(omega: Dict[str, Dict[int, float]], measured: Dict[str, Dict[int, float]]) -> List[dict]:
    """Rows (tensor, b, Omega, measured dL) for tables / plots / rank correlation."""
    rows = []
    for name, tab in omega.items():
        for b, om in tab.items():
            rows.append({"tensor": name, "bits": b, "omega": om, "measured_dL": measured.get(name, {}).get(b)})
    return rows
