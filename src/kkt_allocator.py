"""Mixed-precision bit allocation via Hessian sensitivity and Karush-Kuhn-Tucker (KKT) on VU13P."""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from src.hardware import estimate_hardware_resources, layer_input_bits
from src.models import QUANTIZER_INTEGER_BITS

MIN_TRACE = 1e-8


def compute_hutchinson_hessian_trace(
    model: tf.keras.Model,
    loss_fn: tf.keras.losses.Loss,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    num_hvp: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate the full trace of each layer's Hessian block with Hutchinson's algorithm.

    Tr(H_ll) ~ (1 / M) * sum_j [ z_j^T H_ll z_j ],  z_j Rademacher (entries in {-1, +1}).

    Each layer is probed separately (one Hessian-vector product per layer and draw). Probing all
    layers with one joint vector adds the cross-layer terms z_l^T H_lm z_m, which average to zero
    but inflate the variance, and the estimate of a small layer can come out negative.

    Args:
        model: Trained tf.keras.Model.
        loss_fn: Loss function (e.g. BinaryCrossentropy).
        x_batch: Sample batch of input features.
        y_batch: Corresponding ground-truth labels.
        num_hvp: Number of Rademacher draws per layer.
        seed: Seed of the local random generator (the global NumPy state is not touched).

    Returns:
        Dictionary mapping layer name to the estimated trace (sum over the layer's kernel entries).
        A non-positive estimate is replaced by MIN_TRACE with a warning, since the sensitivity
        model needs c_l > 0.
    """
    rng = np.random.default_rng(seed)
    target_layers = [l for l in model.layers if hasattr(l, "kernel") and l.kernel is not None]
    traces: Dict[str, float] = {}

    for layer in target_layers:
        var = layer.kernel
        total = 0.0
        for _ in range(num_hvp):
            z = tf.constant(rng.choice([-1.0, 1.0], size=var.shape).astype(np.float32))
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:
                    loss = loss_fn(y_batch, model(x_batch, training=False))
                grad = inner_tape.gradient(loss, var)
                grad_dot_z = tf.reduce_sum(grad * z)
            hvp = outer_tape.gradient(grad_dot_z, var)
            total += float(tf.reduce_sum(z * hvp).numpy())
        traces[layer.name] = total / num_hvp

    for name, value in traces.items():
        if value <= MIN_TRACE:
            warnings.warn(
                f"Hutchinson trace of '{name}' is {value:.3g}; replaced by {MIN_TRACE}. "
                "Increase num_hvp or the batch size.",
                RuntimeWarning,
            )
    return {name: max(MIN_TRACE, value) for name, value in traces.items()}


def compute_layer_c_coefficients(
    model: tf.keras.Model,
    traces: Dict[str, float],
    integer_bits: Union[int, Dict[str, int]] = QUANTIZER_INTEGER_BITS,
) -> Dict[str, float]:
    """Sensitivity c_l of the quantization model Omega_l(b) = c_l * 4^(-b).

    Around the trained point (gradient ~ 0), quantizing the kernel W_l with b bits gives

        Delta L ~ 1/2 * (Tr(H_ll) / n_l) * ||Q_b(W_l) - W_l||^2,

    i.e. the mean curvature of the layer times the total squared error. With uniform rounding of
    step Delta = kappa * 2^(-b), ||Q_b(W_l) - W_l||^2 ~ n_l * Delta^2 / 12, so n_l cancels against
    the full trace returned by compute_hutchinson_hessian_trace:

        c_l = Tr(H_ll) * kappa^2 / 24,    kappa = 2^(integer_bits + 1).

    kappa is the width of the range [-2^I, 2^I) of quantized_bits(b, I, alpha=1), the quantizer
    the model is trained with (src.models), which does not adapt to max|W_l|.

    Args:
        model: Trained model.
        traces: Full Hessian trace per layer (sum over the kernel entries).
        integer_bits: Integer bits I of the weight quantizer, one value or one per layer.

    Returns:
        Dictionary mapping layer name to c_l.
    """
    c_coeffs = {}
    for layer in model.layers:
        if layer.name in traces and hasattr(layer, "kernel") and layer.kernel is not None:
            integer = integer_bits.get(layer.name, QUANTIZER_INTEGER_BITS) if isinstance(integer_bits, dict) else integer_bits
            kappa = 2.0 ** (int(integer) + 1)
            c_coeffs[layer.name] = float(traces[layer.name] * kappa ** 2 / 24.0)
    return c_coeffs


def solve_kkt_bit_allocation(
    layer_names: List[str],
    c_coeffs: List[float],
    a_costs: List[float],
    budget_bops: float,
    b_min: int = 2,
    b_max: int = 8,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Dict[str, Any]:
    """Analytical Karush-Kuhn-Tucker (KKT) water-filling bit allocation under BOPs budget.

    Minimizes: sum_i [ c_i * 4^(-b_i) ]
    Subject to: sum_i [ a_i * b_i ] <= budget_bops, and b_min <= b_i <= b_max

    Stationarity gives: b_i*(lambda) = clip( 0.5 * log2( (c_i * ln 4) / (lambda * a_i) ), b_min, b_max )
    """
    c = np.array(c_coeffs, dtype=np.float64)
    a = np.array(a_costs, dtype=np.float64)
    ln4 = np.log(4.0)

    min_bops = np.sum(a * b_min)
    max_bops = np.sum(a * b_max)

    if budget_bops >= max_bops:
        continuous_b = np.full_like(a, b_max)
        integer_b = np.full_like(a, b_max, dtype=int)
        return {
            "lambda_optimal": 0.0,
            "continuous_bits": {name: float(b) for name, b in zip(layer_names, continuous_b)},
            "integer_bits": {name: int(b) for name, b in zip(layer_names, integer_b)},
            "achieved_bops": float(np.sum(a * integer_b)),
            "budget_bops": budget_bops,
            "budget_active": False,
        }

    if budget_bops < min_bops:
        raise ValueError(
            f"Budget {budget_bops} is strictly lower than minimum viable hardware cost {min_bops} (all layers at {b_min} bits)."
        )

    def get_bits(lam: float) -> np.ndarray:
        if lam <= 0:
            return np.full_like(a, b_max)
        raw_b = 0.5 * np.log2((c * ln4) / (lam * a + 1e-12))
        return np.clip(raw_b, b_min, b_max)

    def h_budget(lam: float) -> float:
        return float(np.sum(a * get_bits(lam)) - budget_bops)

    lam_low = 1e-9
    lam_high = 1e4

    while h_budget(lam_low) < 0:
        lam_low /= 10.0
    while h_budget(lam_high) > 0:
        lam_high *= 10.0

    lam_mid = 0.5 * (lam_low + lam_high)
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_low + lam_high)
        val = h_budget(lam_mid)
        if abs(val) < tol:
            break
        if val > 0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    optimal_continuous_b = get_bits(lam_mid)

    floor_b = np.floor(optimal_continuous_b).astype(int)
    floor_b = np.clip(floor_b, b_min, b_max)
    current_bops = np.sum(a * floor_b)

    integer_b = floor_b.copy()
    while True:
        best_layer = -1
        best_efficiency = -1.0
        for i in range(len(integer_b)):
            if integer_b[i] < b_max and (current_bops + a[i] <= budget_bops):
                delta_loss = c[i] * (4.0 ** (-integer_b[i]) - 4.0 ** (-(integer_b[i] + 1)))
                efficiency = delta_loss / (a[i] + 1e-8)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_layer = i

        if best_layer != -1:
            integer_b[best_layer] += 1
            current_bops += a[best_layer]
        else:
            break

    return {
        "lambda_optimal": float(lam_mid),
        "continuous_bits": {name: float(b) for name, b in zip(layer_names, optimal_continuous_b)},
        "integer_bits": {name: int(b) for name, b in zip(layer_names, integer_b)},
        "achieved_bops": float(np.sum(a * integer_b)),
        "budget_bops": budget_bops,
        "budget_active": True,
    }


def check_kkt_stopping_criteria(
    sp_history: List[float],
    g_val: Dict[str, float],
    mu: Dict[str, float],
    tol_sp: float = 1e-4,
    tol_kkt: float = 1e-3,
    window: int = 3,
    stationarity_residual: Optional[float] = None,
    tol_stat: float = 1e-6,
) -> Tuple[bool, Dict[str, Any]]:
    """Checks KKT convergence criteria: SP stabilization, stationarity, feasibility and complementary slackness.

    Conditions at optimal point (x*, mu*):
    1. SP-Index Stabilization: |SP[-1] - mean(SP[-window-1:-1])| < tol_sp
    2. Stationarity: x* minimizes the Lagrangian for mu*. ``stationarity_residual`` is the residual
       of dL/db reported by compute_primal_bit_update for the current configuration; a
       configuration that did not come from that minimization has no residual (None) and is
       never certified.
    3. Primal Feasibility: g_j(x*) <= 0 for all j in {'DSP', 'LUT', 'FF', 'BRAM_bits'}
    4. Dual Feasibility: mu_j* >= 0 for all j
    5. Complementary Slackness: |mu_j* * g_j(x*)| < tol_kkt for all j
    """
    sp_converged = False
    sp_change = 1.0
    if len(sp_history) > window:
        sp_change = float(abs(sp_history[-1] - np.mean(sp_history[-(window + 1):-1])))
        if sp_change < tol_sp:
            sp_converged = True

    stationary = stationarity_residual is not None and float(stationarity_residual) <= tol_stat
    primal_feasible = all(float(g) <= 1e-9 for g in g_val.values())
    dual_feasible = all(float(m) >= -1e-12 for m in mu.values())
    slackness_products = {k: float(abs(mu[k] * g_val[k])) for k in mu.keys()}
    complementary_slackness = all(val < tol_kkt for val in slackness_products.values())

    converged = bool(
        sp_converged and stationary and primal_feasible and dual_feasible and complementary_slackness
    )
    diagnostics = {
        "converged": converged,
        "sp_converged": sp_converged,
        "sp_change": sp_change,
        "stationary": stationary,
        "stationarity_residual": None if stationarity_residual is None else float(stationarity_residual),
        "primal_feasible": primal_feasible,
        "dual_feasible": dual_feasible,
        "complementary_slackness": complementary_slackness,
        "slackness_products": slackness_products,
    }
    return converged, diagnostics


def _lower_convex_hull(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lower convex hull of points (xs[k], ys[k]) with strictly increasing xs.

    Returns the vertices (hull_x, hull_y) of the largest piecewise-linear convex function
    f_hat(x) satisfying f_hat(xs[k]) <= ys[k] for every k. For a linear input curve, this
    reduces to the two endpoints [xs[0], xs[-1]].
    """
    hull_x: List[float] = []
    hull_y: List[float] = []
    for x, y in zip(xs, ys):
        xf, yf = float(x), float(y)
        while len(hull_x) >= 2:
            x1, y1 = hull_x[-2], hull_y[-2]
            x2, y2 = hull_x[-1], hull_y[-1]
            # Pop middle point if slope (1->2) >= slope (2->3) (collinear or concave)
            if (x2 - x1) * (yf - y1) - (y2 - y1) * (xf - x1) <= 1e-12:
                hull_x.pop()
                hull_y.pop()
            else:
                break
        hull_x.append(xf)
        hull_y.append(yf)
    return np.asarray(hull_x, dtype=np.float64), np.asarray(hull_y, dtype=np.float64)


def _piecewise_water_filling(
    c: np.ndarray,
    slopes: np.ndarray,
    b_min: float,
    b_max: float,
) -> Tuple[np.ndarray, float]:
    """Exact minimizer of sum_l [ c_l * 4^(-b_l) + phi_l(b_l) ] over [b_min, b_max]^L.

    ``slopes`` has shape (L, M) where M = int(b_max - b_min) unit intervals
    [b_min + k, b_min + k + 1], and ``slopes[l, k]`` is non-decreasing in k (convex phi_l).

    Returns:
        (optimal continuous bits b*, maximum KKT subdifferential stationarity residual).
    """
    ln4 = np.log(4.0)
    n_layers, n_seg = slopes.shape
    knots = b_min + np.arange(n_seg + 1, dtype=np.float64)
    b = np.empty(n_layers, dtype=np.float64)
    max_res = 0.0

    for l in range(n_layers):
        cl = float(c[l])
        pi = slopes[l]
        d_knots = cl * ln4 * (4.0 ** (-knots))  # strictly decreasing

        if d_knots[0] <= pi[0]:
            bl = b_min
        elif d_knots[-1] >= pi[-1]:
            bl = b_max
        else:
            bl = b_max
            for k in range(n_seg):
                if d_knots[k + 1] <= pi[k]:
                    if d_knots[k] >= pi[k]:
                        bl = float(0.5 * np.log2((cl * ln4) / max(float(pi[k]), 1e-300)))
                        bl = float(np.clip(bl, knots[k], knots[k + 1]))
                    else:
                        bl = float(knots[k])
                    break
        b[l] = bl

        # Exact subdifferential stationarity residual: dist(0, [-d(bl) + pi^-, -d(bl) + pi^+])
        d_bl = cl * ln4 * (4.0 ** (-bl))
        if bl <= b_min + 1e-12:
            g_low, g_high = -np.inf, -d_bl + float(pi[0])
        elif bl >= b_max - 1e-12:
            g_low, g_high = -d_bl + float(pi[-1]), np.inf
        else:
            pos = bl - b_min
            k_near = int(np.rint(pos))
            if abs(pos - k_near) <= 1e-12 and 0 < k_near < n_seg:
                g_low = -d_bl + float(pi[k_near - 1])
                g_high = -d_bl + float(pi[k_near])
            else:
                seg = min(max(int(np.floor(pos)), 0), n_seg - 1)
                g_low = g_high = -d_bl + float(pi[seg])
        res_l = max(0.0, g_low, -g_high)
        if res_l > max_res:
            max_res = float(res_l)

    return b, max_res


def _eval_envelope_constraints(
    b: np.ndarray,
    env_vals: np.ndarray,
    env_slopes: np.ndarray,
    g0: np.ndarray,
    b_min: float,
) -> np.ndarray:
    """Evaluates normalized envelope constraints g_hat_j(b) = g0_j + sum_l f_hat_{j,l}(b_l)."""
    n_res = len(g0)
    if n_res == 0:
        return np.zeros(0, dtype=np.float64)
    n_seg = env_slopes.shape[2]
    pos = np.clip(b - b_min, 0.0, float(n_seg))
    idx = np.minimum(np.floor(pos).astype(int), n_seg - 1)
    frac = pos - idx
    layers_idx = np.arange(b.size)
    # env_vals[:, l, idx[l]] + frac[l] * env_slopes[:, l, idx[l]]
    contrib = env_vals[:, layers_idx, idx] + env_slopes[:, layers_idx, idx] * frac[None, :]
    return g0 + contrib.sum(axis=1)


def solve_relaxed_dual_envelope(
    c: np.ndarray,
    env_vals: np.ndarray,
    g0: np.ndarray,
    b_min: int,
    b_max: int,
    mu0: float = 1e-3,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> Dict[str, Any]:
    """Lagrangian dual of  min_b sum_l c_l 4^(-b_l)  s.t.  g0_j + sum_l f_hat_{j,l}(b_l) <= 0.

    Each ``env_vals[j, l, k]`` is the normalized lower-convex-hull increment of resource j, layer l
    at integer bitwidth b_min + k (with env_vals[j, l, 0] == 0). Because every f_hat_{j,l} is
    convex and piecewise-linear on [b_min, b_max] and Omega(b) is strictly convex:

    * For any mu >= 0, the Lagrangian separates by layer and has a unique exact minimizer
      b*(mu) via piecewise water-filling (_piecewise_water_filling), satisfying KKT stationarity
      (0 in dL/db_l) by construction.
    * The dual function q(mu) = L(b*(mu), mu) is concave and continuously differentiable with
      gradient g_hat(b*(mu)).
    * Projected ascent in log2(mu) with per-coordinate sign-flip damping finds mu* >= 0 satisfying
      primal feasibility and complementary slackness, certifying global KKT optimality on the
      continuous convex relaxation and a valid weak-duality lower bound (duality_gap >= 0).
    """
    c = np.asarray(c, dtype=np.float64)
    env_vals = np.asarray(env_vals, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    n_res = len(g0)
    b_min_f, b_max_f = float(b_min), float(b_max)
    n_seg = int(b_max - b_min)

    if n_res == 0:
        b = np.full(c.size, b_max_f, dtype=np.float64)
        omega = float(np.sum(c * 4.0 ** (-b)))
        return {
            "bits": b,
            "mu": np.zeros(0, dtype=np.float64),
            "g": np.zeros(0, dtype=np.float64),
            "omega": omega,
            "dual_value": omega,
            "stationarity_residual": 0.0,
            "complementary_slackness": 0.0,
            "iterations": 1,
            "converged": True,
        }

    env_slopes = np.diff(env_vals, axis=2)  # shape (n_res, n_layers, n_seg)

    if np.any(g0 > 1e-12):
        raise ValueError("Infeasible budget: violated even with every layer at b_min.")

    # Check if all constraints are already inactive at b_max -> mu* = 0 immediately
    g_at_max = g0 + env_vals[:, :, -1].sum(axis=1)
    if np.all(g_at_max <= tol):
        b = np.full(c.size, b_max_f, dtype=np.float64)
        omega = float(np.sum(c * 4.0 ** (-b)))
        mu_zero = np.zeros(n_res, dtype=np.float64)
        return {
            "bits": b,
            "mu": mu_zero,
            "g": g_at_max,
            "omega": omega,
            "dual_value": omega,
            "stationarity_residual": 0.0,
            "complementary_slackness": 0.0,
            "iterations": 1,
            "converged": True,
        }

    mu = np.full(n_res, float(mu0), dtype=np.float64)
    # Conservative initial step per constraint based on steepest envelope slope sum
    slope_scale = np.maximum(np.maximum(env_slopes, 0.0).max(axis=2).sum(axis=1), 1e-12)
    eta_max = 1.0 / slope_scale
    eta = 0.5 * eta_max
    prev_g: Optional[np.ndarray] = None
    converged = False
    iteration = 1

    for iteration in range(1, max_iter + 1):
        combined_slopes = np.tensordot(mu, env_slopes, axes=(0, 0))  # (n_layers, n_seg)
        b, residual = _piecewise_water_filling(c, combined_slopes, b_min_f, b_max_f)
        g = _eval_envelope_constraints(b, env_vals, env_slopes, g0, b_min_f)
        omega = float(np.sum(c * 4.0 ** (-b)))
        if g.max() < tol and np.max(np.abs(mu * g)) < tol * max(omega, 1e-12):
            converged = True
            break

        if prev_g is not None:
            flipped = (g * prev_g) < 0.0
            eta = np.where(flipped, np.maximum(eta * 0.5, 1e-12), np.minimum(eta * 1.05, eta_max))
        prev_g = g.copy()

        step = np.clip(2.0 * eta * np.clip(g, -2.0, 2.0), -8.0, 8.0)
        mu = np.clip(mu * (2.0 ** step), 1e-300, 1e100)

    # Zero out multipliers of constraints with slack and recompute exact minimizer
    if np.any(g < -tol):
        mu = np.where(g < -tol, 0.0, mu)
        combined_slopes = np.tensordot(mu, env_slopes, axes=(0, 0))
        b, residual = _piecewise_water_filling(c, combined_slopes, b_min_f, b_max_f)
        g = _eval_envelope_constraints(b, env_vals, env_slopes, g0, b_min_f)
        omega = float(np.sum(c * 4.0 ** (-b)))

    return {
        "bits": b,
        "mu": mu,
        "g": g,
        "omega": omega,
        "dual_value": omega + float(mu @ g),
        "stationarity_residual": float(residual),
        "complementary_slackness": float(np.max(np.abs(mu * g), initial=0.0)),
        "iterations": int(iteration),
        "converged": bool(converged),
    }


def solve_relaxed_dual(
    c: np.ndarray,
    K: np.ndarray,
    g0: np.ndarray,
    b_min: float,
    b_max: float,
    mu0: float = 1e-3,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> Dict[str, Any]:
    """Lagrangian dual of  min_b sum_l c_l 4^(-b_l)  s.t.  K b + g0 <= 0,  b_min <= b_l <= b_max.

    Special case of ``solve_relaxed_dual_envelope`` where every resource constraint is affine in b.
    """
    c = np.asarray(c, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    n_res = len(g0)
    b_min_i, b_max_i = int(round(b_min)), int(round(b_max))
    if n_res == 0:
        return solve_relaxed_dual_envelope(
            c, np.zeros((0, c.size, max(b_max_i - b_min_i, 1) + 1)), g0, b_min_i, b_max_i,
            mu0=mu0, tol=tol, max_iter=max_iter,
        )
    deltas = np.arange(b_max_i - b_min_i + 1, dtype=np.float64)  # 0 .. (b_max - b_min)
    env_vals = K[:, :, None] * deltas[None, None, :]
    g0_shifted = K @ np.full(c.size, float(b_min_i)) + g0
    return solve_relaxed_dual_envelope(
        c, env_vals, g0_shifted, b_min_i, b_max_i, mu0=mu0, tol=tol, max_iter=max_iter
    )


class LagrangeKKTQuantizerOptimizer:
    """Closed-Loop Primal-Dual optimizer with Lagrange Multipliers and KKT conditions on VU13P.

    Implements the 3-step loop:
    1. QUANTIZED NEURAL NETWORK (QKeras): Configures bitwidths and architecture (w_l, a_l, n_l).
    2. PERFORMANCE & COST EVALUATION: Computes SP-Index loss (1 - SP) and VU13P resources.
    3. LAGRANGE / KKT OPTIMIZATION CORE: Evaluates g_j(x) <= 0, updates shadow prices (mu_j),
       and computes the next primal bitwidth allocation (w_l^{(t+1)}, a_l^{(t+1)}).
    """

    def __init__(
        self,
        hardware_limits: Dict[str, float],
        lr_x: float = 0.05,
        lr_mu: float = 0.50,
        normalize_constraints: bool = True,
    ):
        self.limits = {k: float(v) for k, v in hardware_limits.items()}
        self.lr_x = lr_x
        self.lr_mu = lr_mu
        self.normalize_constraints = normalize_constraints

        # Dual variables (shadow prices) mu_j >= 0
        self.mu: Dict[str, float] = {k: 0.0 for k in self.limits.keys()}
        self.history: List[Dict[str, Any]] = []

    def evaluate_constraints(
        self, resource_usage: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Computes raw g_j(x) = R_j - B_j and normalized g_norm_j(x) = (R_j - B_j) / B_j."""
        g_raw = {}
        g_norm = {}
        for k, limit_val in self.limits.items():
            used = float(resource_usage.get(k, 0.0))
            diff = used - limit_val
            g_raw[k] = diff
            denom = max(limit_val, 1.0)
            g_norm[k] = diff / denom
        return g_raw, g_norm

    def lagrangian_penalty(
        self, sp_loss: float, resource_usage: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Computes the Lagrangian L(x, mu) = (1 - SP) + sum_j mu_j * g_j(x)."""
        g_raw, g_norm = self.evaluate_constraints(resource_usage)
        g_used = g_norm if self.normalize_constraints else g_raw

        penalty = 0.0
        for k in self.limits.keys():
            penalty += self.mu[k] * g_used[k]

        total_cost = float(sp_loss + penalty)
        return total_cost, g_used

    def update_multipliers(self, g_val: Dict[str, float]) -> Dict[str, float]:
        """Projected subgradient ascent on the dual: mu_j <- max(0, mu_j + lr_mu * g_j(x)).

        A constraint with slack (g_j < 0) lowers its price and reaches mu_j = 0 only through this
        projection, which is how complementary slackness mu_j * g_j = 0 arises at the optimum.
        """
        for k in self.mu.keys():
            self.mu[k] = float(max(0.0, self.mu[k] + self.lr_mu * float(g_val[k])))
        return self.mu.copy()

    def compute_primal_bit_update(
        self,
        layer_names: List[str],
        c_coeffs: List[float],
        macs_list: List[float],
        params_list: List[float],
        current_w: Dict[str, int],
        current_a: Dict[str, int],
        b_min: int = 2,
        b_max: int = 8,
        input_bits: int = 8,
        cost_fn: Optional[Callable[[Dict[str, int], Dict[str, int]], Dict[str, float]]] = None,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
        """Next weight bits from the KKT convex-envelope relaxation of min sum_l c_l 4^(-w_l) s.t. g_j(w) <= 0.

        ``current_a`` is keyed by the layer whose ReLU output is quantized; the cost of a weight bit
        of layer l uses the precision entering l (layer_input_bits). Steps:

        1. With B_DSP = 0, weights and activations are limited to 8 bits (above that the cost model
           maps multiplications to DSP48E2).
        2. For each layer l and resource j, evaluates the incremental cost curve f_{j,l}(w) over
           w in {b_min, ..., w_max} and builds its lower convex hull f_hat_{j,l}(w) <= f_{j,l}(w).
           This is exact for linear cost models and forms a valid convex piecewise-linear
           relaxation of the calibrated CSD synthesis model.
        3. Solves the continuous convex relaxation via dual ascent (solve_relaxed_dual_envelope),
           giving KKT-optimal continuous bits b*(mu*), shadow prices mu* (stored in ``self.mu``),
           exact subdifferential stationarity and a valid weak-duality lower bound.
        4. Projects b*(mu*) to integer bits under the actual cost model via KKT-guided floor,
           feasibility repair, marginal-efficiency greedy fill and O(L^2) swap refinement.

        ``cost_fn(weights_bits, act_bits) -> {resource: usage}`` is the cost model; use
        src.hardware.make_cost_fn(reference_model, input_bits) for the calibrated VU13P model.
        When omitted, the analytical estimate_hardware_resources is used with ``input_bits``.
        ``current_w`` is not needed by this step and is kept for the call signature.

        Returns:
            (next weight bits, activation bits used, diagnostics with the continuous KKT solution,
            shadow prices, KKT residuals, valid dual bound, non-negative duality gap and integer
            allocation).
        """
        dsp_forbidden = self.limits.get("DSP", 0.0) <= 0.0
        w_max = min(b_max, 8) if dsp_forbidden else b_max
        next_a = {k: (min(int(v), 8) if dsp_forbidden else int(v)) for k, v in current_a.items()}
        a_in = layer_input_bits(layer_names, next_a, input_bits)
        c = np.asarray(c_coeffs, dtype=np.float64)
        n_layers = len(layer_names)

        def resources(w_bits) -> Dict[str, float]:
            if cost_fn is not None:
                return cost_fn(dict(zip(layer_names, (int(w) for w in w_bits))), next_a)
            return estimate_hardware_resources([
                {"w_bits": int(w), "a_bits": a_in[name], "params": float(p), "macc": float(m)}
                for name, w, p, m in zip(layer_names, w_bits, params_list, macs_list)
            ])

        names = list(self.limits.keys())
        n_res = len(names)
        scale = np.array([max(self.limits[r], 1.0) for r in names], dtype=np.float64)
        budget = np.array([self.limits[r] for r in names], dtype=np.float64)

        # Sample separable per-layer incremental curves and construct lower convex hulls
        grid = np.arange(b_min, w_max + 1, dtype=np.float64)
        n_grid = len(grid)
        base = resources([b_min] * n_layers)
        base_vec = np.array([float(base[r]) for r in names], dtype=np.float64)
        g0_all = (base_vec - budget) / scale

        raw_inc = np.zeros((n_res, n_layers, n_grid), dtype=np.float64)
        env_vals = np.zeros((n_res, n_layers, n_grid), dtype=np.float64)
        for l in range(n_layers):
            for idx_w, w_val in enumerate(range(b_min + 1, w_max + 1), start=1):
                w_probe = [b_min] * n_layers
                w_probe[l] = w_val
                probe_res = resources(w_probe)
                for j, r in enumerate(names):
                    raw_inc[j, l, idx_w] = (float(probe_res[r]) - base_vec[j]) / scale[j]
            for j in range(n_res):
                hx, hy = _lower_convex_hull(grid, raw_inc[j, l])
                env_vals[j, l] = np.interp(grid, hx, hy)

        # Constraints that do not vary with w are either always satisfied (price 0) or infeasible
        varies = np.any(np.abs(env_vals) > 1e-15, axis=(1, 2))
        if np.any(~varies & (g0_all > 1e-12)):
            bad = [r for r, v, g in zip(names, varies, g0_all) if not v and g > 1e-12]
            raise ValueError(f"Infeasible budget for {bad}: not reachable by changing weight bits.")
        rows = [j for j in range(n_res) if varies[j]]
        dual = solve_relaxed_dual_envelope(c, env_vals[rows], g0_all[rows], b_min, w_max)

        self.mu = {r: 0.0 for r in names}
        mu_all = np.zeros(n_res, dtype=np.float64)
        for j, m in zip(rows, dual["mu"]):
            self.mu[names[j]] = float(m)
            mu_all[j] = float(m)

        def omega(w) -> float:
            return float(np.sum(c * 4.0 ** (-np.asarray(w, dtype=np.float64))))

        def is_feasible(res: Dict[str, float]) -> bool:
            return all(float(res[r]) <= self.limits[r] + 1e-9 for r in names)

        # KKT-guided integer projection from continuous b*(mu*) under the actual cost model
        w_int = np.clip(np.floor(dual["bits"] + 1e-9).astype(int), b_min, w_max)
        res_int = resources(w_int)

        # 1. Feasibility repair if a non-convex bump above the hull caused an overshoot
        while not is_feasible(res_int):
            best_l, best_ratio = -1, float("inf")
            for l in range(n_layers):
                if w_int[l] <= b_min:
                    continue
                cand = w_int.copy()
                cand[l] -= 1
                cand_res = resources(cand)
                relief = sum(
                    max(0.0, float(res_int[r]) - float(cand_res[r])) / scale[j]
                    for j, r in enumerate(names)
                    if float(res_int[r]) > self.limits[r] + 1e-9
                )
                if relief > 1e-12:
                    loss_inc = float(c[l] * (4.0 ** (-cand[l]) - 4.0 ** (-w_int[l])))
                    ratio = loss_inc / relief
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_l = l
            if best_l == -1:
                raise ValueError("No integer bit allocation in [b_min, b_max] satisfies the budget.")
            w_int[best_l] -= 1
            res_int = resources(w_int)

        def greedy_fill(w_vec: np.ndarray, cur_res: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
            w_cur = w_vec.copy()
            r_cur = cur_res
            weights = mu_all if np.any(mu_all > 0.0) else np.ones(n_res, dtype=np.float64)
            while True:
                best_l, best_eff = -1, -1.0
                best_res = r_cur
                for l in range(n_layers):
                    if w_cur[l] >= w_max:
                        continue
                    cand = w_cur.copy()
                    cand[l] += 1
                    cand_res = resources(cand)
                    if not is_feasible(cand_res):
                        continue
                    delta_loss = float(c[l] * (4.0 ** (-w_cur[l]) - 4.0 ** (-cand[l])))
                    weighted_cost = sum(
                        weights[j] * max(0.0, float(cand_res[r]) - float(r_cur[r])) / scale[j]
                        for j, r in enumerate(names)
                    )
                    eff = delta_loss / max(weighted_cost, 1e-12)
                    if eff > best_eff:
                        best_eff = eff
                        best_l = l
                        best_res = cand_res
                if best_l == -1:
                    break
                w_cur[best_l] += 1
                r_cur = best_res
            return w_cur, r_cur

        # 2. Marginal-efficiency greedy fill guided by KKT shadow prices mu*
        w_int, res_int = greedy_fill(w_int, res_int)

        # 3. O(L^2) local swap refinement around the KKT neighborhood (handles integer ties)
        improved = True
        while improved:
            improved = False
            best_w, best_res, best_om = w_int, res_int, omega(w_int)
            for i in range(n_layers):
                if w_int[i] <= b_min:
                    continue
                for j in range(n_layers):
                    if i == j or w_int[j] >= w_max:
                        continue
                    cand = w_int.copy()
                    cand[i] -= 1
                    cand[j] += 1
                    cand_res = resources(cand)
                    if not is_feasible(cand_res):
                        continue
                    cand, cand_res = greedy_fill(cand, cand_res)
                    cand_om = omega(cand)
                    if cand_om < best_om - 1e-12:
                        best_w, best_res, best_om = cand, cand_res, cand_om
                        improved = True
            w_int, res_int = best_w, best_res

        env_slopes_all = np.diff(env_vals, axis=2)
        g_cont = _eval_envelope_constraints(dual["bits"], env_vals, env_slopes_all, g0_all, float(b_min))
        omega_int = omega(w_int)

        info = {
            "continuous_bits": {n: float(b) for n, b in zip(layer_names, dual["bits"])},
            "integer_bits": {n: int(w) for n, w in zip(layer_names, w_int)},
            "activation_bits": dict(next_a),
            "activation_input_bits": dict(a_in),
            "shadow_prices": dict(self.mu),
            "g_continuous": {r: float(g) for r, g in zip(names, g_cont)},
            "stationarity_residual": dual["stationarity_residual"],
            "complementary_slackness": dual["complementary_slackness"],
            "dual_iterations": int(dual["iterations"]),
            "dual_converged": bool(dual["converged"]),
            "dual_bound": float(dual["dual_value"]),
            "omega_continuous": float(dual["omega"]),
            "omega_integer": omega_int,
            "duality_gap": omega_int - float(dual["dual_value"]),
            "integer_resources": {r: float(v) for r, v in res_int.items()},
        }
        return info["integer_bits"], next_a, info

    def record_step(
        self,
        iteration: int,
        sp_val: float,
        w_bits: Dict[str, int],
        a_bits: Dict[str, int],
        resource_usage: Dict[str, float],
        g_val: Dict[str, float],
        lagrangian_cost: float,
        kkt_diag: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records full primal-dual telemetry for plotting and article tables."""
        g_raw, _ = self.evaluate_constraints(resource_usage)
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "sp_val": float(sp_val),
            "sp_loss": float(1.0 - sp_val),
            "lagrangian_cost": float(lagrangian_cost),
            "w_bits": dict(w_bits),
            "a_bits": dict(a_bits),
            "resources": {k: float(v) for k, v in resource_usage.items()},
            "g_raw": {k: float(v) for k, v in g_raw.items()},
            "g_val": {k: float(v) for k, v in g_val.items()},
            "mu": {k: float(v) for k, v in self.mu.items()},
            "kkt_diagnostics": kkt_diag,
        }
        if extra:
            entry.update(extra)
        self.history.append(entry)
