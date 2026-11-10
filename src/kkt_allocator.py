"""Mixed-precision bit allocation via Hessian sensitivity and Karush-Kuhn-Tucker (KKT) on VU13P."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf


def compute_hutchinson_hessian_trace(
    model: tf.keras.Model,
    loss_fn: tf.keras.losses.Loss,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    num_hvp: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate the trace of the Hessian for each layer's weights using Hutchinson's algorithm.

    Tr(H_i) ~ (1 / M) * sum_j [ z_j^T * H_i * z_j ]
    where z_j is a random Rademacher vector (elements in {-1, +1}).

    Args:
        model: Trained tf.keras.Model.
        loss_fn: Loss function (e.g. BinaryCrossentropy).
        x_batch: Sample batch of input features.
        y_batch: Corresponding ground-truth labels.
        num_hvp: Number of Hessian-Vector Products (Monte Carlo evaluations).
        seed: Random seed for Rademacher vectors.

    Returns:
        Dictionary mapping layer name to its estimated Hessian trace Tr(H_i).
    """
    np.random.seed(seed)
    target_layers = [l for l in model.layers if hasattr(l, "kernel") and l.kernel is not None]
    trainable_vars = [l.kernel for l in target_layers]
    layer_names = [l.name for l in target_layers]
    var_traces = {name: 0.0 for name in layer_names}

    for _ in range(num_hvp):
        v_list = []
        for var in trainable_vars:
            rademacher = np.random.choice([-1.0, 1.0], size=var.shape).astype(np.float32)
            v_list.append(tf.convert_to_tensor(rademacher))

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                y_pred = model(x_batch, training=False)
                loss = loss_fn(y_batch, y_pred)

            grads = inner_tape.gradient(loss, trainable_vars)
            gv_products = [tf.reduce_sum(g * v) for g, v in zip(grads, v_list) if g is not None]
            total_gv = tf.add_n(gv_products)

        hvps = outer_tape.gradient(total_gv, trainable_vars)

        for name, v, hvp in zip(layer_names, v_list, hvps):
            if hvp is not None:
                quad_form = tf.reduce_sum(v * hvp).numpy()
                var_traces[name] += float(quad_form) / num_hvp

    return {k: max(1e-8, v) for k, v in var_traces.items()}


def compute_layer_c_coefficients(
    model: tf.keras.Model,
    traces: Dict[str, float],
) -> Dict[str, float]:
    """Compute perturbation sensitivity coefficients: c_i = Tr(H_i) * (n_i * r_i^2) / 3.

    Args:
        model: Trained model.
        traces: Estimated Hessian trace per layer.

    Returns:
        Dictionary mapping layer name to c_i coefficient.
    """
    c_coeffs = {}
    for layer in model.layers:
        if layer.name in traces and hasattr(layer, "kernel") and layer.kernel is not None:
            weights = layer.kernel.numpy()
            n_i = weights.size
            r_i = float(np.max(np.abs(weights))) + 1e-6
            tr_h = traces[layer.name]
            c_i = tr_h * (n_i * (r_i ** 2)) / 3.0
            c_coeffs[layer.name] = float(c_i)
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
) -> Tuple[bool, Dict[str, Any]]:
    """Checks KKT convergence criteria: SP stabilization, Primal Feasibility, and Complementary Slackness.

    Conditions at optimal point (x*, mu*):
    1. SP-Index Stabilization: |SP[-1] - mean(SP[-window-1:-1])| < tol_sp
    2. Primal Feasibility: g_j(x*) <= 0 for all j in {'DSP', 'LUT', 'FF', 'BRAM_bits'}
    3. Dual Feasibility: mu_j* >= 0 for all j
    4. Complementary Slackness: |mu_j* * g_j(x*)| < tol_kkt for all j
    """
    sp_converged = False
    sp_change = 1.0
    if len(sp_history) > window:
        sp_change = float(abs(sp_history[-1] - np.mean(sp_history[-(window + 1):-1])))
        if sp_change < tol_sp:
            sp_converged = True

    primal_feasible = all(float(g) <= 1e-9 for g in g_val.values())
    dual_feasible = all(float(m) >= -1e-12 for m in mu.values())
    slackness_products = {k: float(abs(mu[k] * g_val[k])) for k in mu.keys()}
    complementary_slackness = all(val < tol_kkt for val in slackness_products.values())

    converged = bool(sp_converged and primal_feasible and dual_feasible and complementary_slackness)
    diagnostics = {
        "converged": converged,
        "sp_converged": sp_converged,
        "sp_change": sp_change,
        "primal_feasible": primal_feasible,
        "dual_feasible": dual_feasible,
        "complementary_slackness": complementary_slackness,
        "slackness_products": slackness_products,
    }
    return converged, diagnostics


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
        slack_decay: float = 0.30,
    ):
        self.limits = {k: float(v) for k, v in hardware_limits.items()}
        self.lr_x = lr_x
        self.lr_mu = lr_mu
        self.normalize_constraints = normalize_constraints
        self.slack_decay = slack_decay

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
        """Computes the penalized Lagrangian cost L(x, mu) = (1 - SP) + sum_j mu_j * max(0, g_j(x))."""
        g_raw, g_norm = self.evaluate_constraints(resource_usage)
        g_used = g_norm if self.normalize_constraints else g_raw

        penalty = 0.0
        for k in self.limits.keys():
            penalty += self.mu[k] * max(0.0, g_used[k])

        total_cost = float(sp_loss + penalty)
        return total_cost, g_used

    def update_multipliers(self, g_val: Dict[str, float]) -> Dict[str, float]:
        """Dual update via projected subgradient ascent + complementary slackness projection.

        - If g_j(x) > 0 (resource limit violated): increases shadow price mu_j^{(t+1)} = mu_j^{(t)} + lr_mu * g_j.
        - If g_j(x) <= 0 (resource feasible with slack): decays mu_j toward 0 so that
          complementary slackness (mu_j* * g_j(x*) = 0) is strictly satisfied for inactive constraints.
        """
        for k in self.mu.keys():
            gv = float(g_val[k])
            if gv > 0.0:
                self.mu[k] = float(self.mu[k] + self.lr_mu * gv)
            else:
                updated = self.mu[k] + self.lr_mu * gv
                self.mu[k] = float(max(0.0, min(updated, self.mu[k] * self.slack_decay)))
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
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
        """Computes the next primal configuration (w_l^{(t+1)}, a_l^{(t+1)}) from current shadow prices mu_j.

        Follows KKT stationarity on the VU13P:
        1. If mu_DSP > 0 or B_DSP == 0: enforces sub-byte arithmetic (w_l <= 8, a_l <= 8) so DSP = 0.
        2. If mu_LUT > 0 or mu_BRAM > 0: computes the effective BOPs/LUT budget from B_LUT and B_BRAM
           and solves the analytical KKT water-filling problem weighted by layer sensitivities c_l.
        3. Adapts activation bitwidths a_l on high-MAC layers when LUT pressure remains active.
        """
        # 1. Enforce DSP constraint (UltraScale+ uses DSP48E2 when w_l > 8 or a_l > 8)
        eff_b_max = b_max if (self.mu.get("DSP", 0.0) > 0.0 or self.limits.get("DSP", 0.0) <= 0.0) else 16

        # 2. Check if all constraints are already feasible and only DSP was violated at 16-bit
        lut_pressure = self.mu.get("LUT", 0.0)
        bram_pressure = self.mu.get("BRAM_bits", 0.0)
        ff_pressure = self.mu.get("FF", 0.0)

        # If no LUT/BRAM multiplier is active yet (e.g., right after 16-bit DSP violation),
        # step down to uniform 8-bit first so the dual loop can measure pure LUT/BRAM pressure.
        if lut_pressure <= 1e-9 and any(v > 8 for v in current_w.values()):
            next_w = {name: eff_b_max for name in layer_names}
            next_a = {name: eff_b_max for name in current_a.keys()}
            return next_w, next_a, {"regime": "sub_byte_8bit_projection"}

        # 3. When LUT or BRAM shadow prices are active, solve KKT water-filling under VU13P LUT/BRAM ceiling
        a_bits_vec = np.array([float(current_a.get(name, 8)) for name in layer_names], dtype=np.float64)
        macs_vec = np.array(macs_list, dtype=np.float64)
        a_costs = (macs_vec * a_bits_vec).tolist()

        # Target BOPs derived from VU13P LUT budget (since LUT = BOPs / 18 in sub-byte mode)
        lut_limit = self.limits.get("LUT", 2500.0)
        bops_from_lut_limit = lut_limit * 18.0

        # Tighten effective budget proportionally to accumulated shadow price mu_LUT + mu_BRAM
        pressure_factor = min(1.0, max(0.58, 1.0 / (1.0 + 0.35 * (lut_pressure + bram_pressure + ff_pressure))))
        bops_8bit = float(np.sum(macs_vec * 8.0 * 8.0))
        target_bops = min(bops_from_lut_limit, bops_8bit * pressure_factor)

        kkt_sol = solve_kkt_bit_allocation(
            layer_names=layer_names,
            c_coeffs=c_coeffs,
            a_costs=a_costs,
            budget_bops=target_bops,
            b_min=b_min,
            b_max=eff_b_max,
        )
        next_w = kkt_sol["integer_bits"]

        # Check resulting LUT usage with current_a; if LUT > lut_limit, quantize dense activation to 6 bits
        est_bops_with_cur_a = sum(
            macs_list[i] * next_w[name] * current_a.get(name, 8)
            for i, name in enumerate(layer_names)
        )
        est_luts = est_bops_with_cur_a / 18.0
        next_a = dict(current_a)
        for k in next_a:
            next_a[k] = min(next_a[k], eff_b_max)

        if est_luts > lut_limit and "dense_1" in next_a:
            next_a["dense_1"] = 6

        return next_w, next_a, kkt_sol

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
