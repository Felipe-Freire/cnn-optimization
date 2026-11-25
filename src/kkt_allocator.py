"""Mixed-precision bit allocation via Hessian sensitivity and Karush-Kuhn-Tucker (KKT) on VU13P."""

import itertools
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    integer_bits: int = QUANTIZER_INTEGER_BITS,
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
        integer_bits: Integer bits I of the weight quantizer.

    Returns:
        Dictionary mapping layer name to c_l.
    """
    kappa = 2.0 ** (integer_bits + 1)
    c_coeffs = {}
    for layer in model.layers:
        if layer.name in traces and hasattr(layer, "kernel") and layer.kernel is not None:
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

    Row j of (K, g0) is the normalized constraint g_j(b) = (R_j(b) - B_j) / max(B_j, 1), linear in b.
    For fixed prices mu >= 0 the Lagrangian separates by layer and its minimizer is the water-filling

        b_l(mu) = clip( 1/2 * log2( c_l ln4 / pi_l ), b_min, b_max ),   pi_l = sum_j mu_j K_jl,

    so stationarity holds by construction. The primal is strictly convex, so the dual function is
    concave and differentiable with gradient g(b(mu)); mu is moved by projected ascent in
    t_j = log2(mu_j), which keeps mu_j >= 0 and changes it by a constant factor per unit of
    violation. It stops when b(mu) is feasible and complementary slackness holds (relative to the
    objective): the four KKT conditions, which are sufficient here.
    """
    c = np.asarray(c, dtype=np.float64)
    if np.any(K @ np.full(c.size, float(b_min)) + g0 > 0):
        raise ValueError("Infeasible budget: violated even with every layer at b_min.")
    mu = np.full(len(g0), float(mu0))
    eta = 1.0 / max(float(K.sum(axis=1).max()), 1e-12) if len(g0) else 0.0
    converged = False
    for iteration in range(1, max_iter + 1):
        price = mu @ K if len(g0) else np.zeros_like(c)
        b = np.clip(0.5 * np.log2(c * np.log(4.0) / np.maximum(price, 1e-300)), b_min, b_max)
        g = K @ b + g0 if len(g0) else np.zeros(0)
        omega = float(np.sum(c * 4.0 ** (-b)))
        if not len(g0) or (g.max() < tol and np.max(np.abs(mu * g)) < tol * omega):
            converged = True
            break
        mu = mu * 2.0 ** (2.0 * eta * np.clip(g, -2.0, 2.0))

    # Prices of constraints with slack only decay geometrically; complementary slackness sets them
    # to zero, and the minimizer is recomputed with the remaining prices.
    if len(g0) and np.any(g < -tol):
        mu = np.where(g < -tol, 0.0, mu)
        price = mu @ K
        b = np.clip(0.5 * np.log2(c * np.log(4.0) / np.maximum(price, 1e-300)), b_min, b_max)
        g = K @ b + g0
        omega = float(np.sum(c * 4.0 ** (-b)))

    # dL/db_l: zero in the interior, >= 0 at b_min (nu_lower) and <= 0 at b_max (nu_upper)
    grad = -c * np.log(4.0) * 4.0 ** (-b) + (mu @ K if len(g0) else 0.0)
    at_min, at_max = b <= b_min + 1e-12, b >= b_max - 1e-12
    interior = ~(at_min | at_max)
    residual = max(
        float(np.max(np.abs(grad[interior]), initial=0.0)),
        float(np.max(-grad[at_min], initial=0.0)),
        float(np.max(grad[at_max], initial=0.0)),
    )
    return {
        "bits": b,
        "mu": mu,
        "g": g,
        "omega": omega,
        "dual_value": omega + float(mu @ g) if len(g0) else omega,
        "stationarity_residual": residual,
        "complementary_slackness": float(np.max(np.abs(mu * g), initial=0.0)),
        "iterations": iteration,
        "converged": converged,
    }


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
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
        """Next weight bits from the Lagrangian of  min sum_l c_l 4^(-w_l)  s.t.  g_j(w) <= 0.

        ``current_a`` is keyed by the layer whose ReLU output is quantized; the cost of a weight bit
        of layer l uses the precision entering l (layer_input_bits). Steps:

        1. With B_DSP = 0, weights and activations are limited to 8 bits (above that the cost model
           maps multiplications to DSP48E2).
        2. Each constraint g_j(w) = (R_j(w) - B_j) / max(B_j, 1) is written as K_j w + g0_j from the
           cost model (exact while the cost is linear in w_l, as in estimate_hardware_resources up
           to 8 bits).
        3. The relaxed problem is solved by dual ascent (solve_relaxed_dual); the shadow prices of
           the solution are stored in ``self.mu``.
        4. The integer bits are the exact optimum over all combinations in [b_min, b_max], checked
           against the cost model itself.

        ``current_w`` is not needed by this step and is kept for the call signature.

        Returns:
            (next weight bits, activation bits used, diagnostics with the continuous solution,
            shadow prices, KKT residuals, dual bound and integer optimum).
        """
        dsp_forbidden = self.limits.get("DSP", 0.0) <= 0.0
        w_max = min(b_max, 8) if dsp_forbidden else b_max
        next_a = {k: (min(int(v), 8) if dsp_forbidden else int(v)) for k, v in current_a.items()}
        a_in = layer_input_bits(layer_names, next_a, input_bits)
        c = np.asarray(c_coeffs, dtype=np.float64)

        def resources(w_bits) -> Dict[str, float]:
            return estimate_hardware_resources([
                {"w_bits": int(w), "a_bits": a_in[name], "params": float(p), "macc": float(m)}
                for name, w, p, m in zip(layer_names, w_bits, params_list, macs_list)
            ])

        names = list(self.limits.keys())
        scale = np.array([max(self.limits[r], 1.0) for r in names])
        budget = np.array([self.limits[r] for r in names])
        base = resources([b_min] * len(layer_names))
        slopes = np.zeros((len(names), len(layer_names)))
        for l in range(len(layer_names)):
            w = [b_min] * len(layer_names)
            w[l] = w_max
            top = resources(w)
            for j, r in enumerate(names):
                slopes[j, l] = (top[r] - base[r]) / max(w_max - b_min, 1)
        K = slopes / scale[:, None]
        g0 = (np.array([base[r] for r in names]) - slopes.sum(axis=1) * b_min - budget) / scale

        # constraints that do not depend on w are either always satisfied (price 0) or infeasible
        varies = np.any(np.abs(K) > 0.0, axis=1)
        if np.any(~varies & (g0 > 1e-12)):
            bad = [r for r, v, g in zip(names, varies, g0) if not v and g > 1e-12]
            raise ValueError(f"Infeasible budget for {bad}: not reachable by changing weight bits.")
        rows = [j for j in range(len(names)) if varies[j]]
        dual = solve_relaxed_dual(c, K[rows], g0[rows], b_min, w_max)
        self.mu = {r: 0.0 for r in names}
        for j, m in zip(rows, dual["mu"]):
            self.mu[names[j]] = float(m)

        def omega(w) -> float:
            return float(np.sum(c * 4.0 ** (-np.asarray(w, dtype=np.float64))))

        feasible = []
        for w in itertools.product(range(b_min, w_max + 1), repeat=len(layer_names)):
            used = resources(w)
            if all(used[r] <= self.limits[r] + 1e-9 for r in names):
                feasible.append(w)
        if not feasible:
            raise ValueError("No integer bit allocation in [b_min, b_max] satisfies the budget.")
        w_int = min(feasible, key=omega)
        g_cont = K @ dual["bits"] + g0

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
            "omega_integer": omega(w_int),
            "duality_gap": omega(w_int) - float(dual["dual_value"]),
            "integer_resources": {r: float(v) for r, v in resources(w_int).items()},
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
    ) -> None:
        """Records full primal-dual telemetry for plotting and article tables."""
        g_raw, _ = self.evaluate_constraints(resource_usage)
        self.history.append({
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
        })
