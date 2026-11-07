"""Mixed-precision bit allocation via Hessian sensitivity and Karush-Kuhn-Tucker (KKT)."""

from typing import Dict, List, Optional, Tuple
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
    # Identify layers that contain a trainable kernel weight
    target_layers = [l for l in model.layers if hasattr(l, "kernel") and l.kernel is not None]
    trainable_vars = [l.kernel for l in target_layers]
    layer_names = [l.name for l in target_layers]
    var_traces = {name: 0.0 for name in layer_names}

    for _ in range(num_hvp):
        # Generate random Rademacher vector for each trainable kernel
        v_list = []
        for var in trainable_vars:
            rademacher = np.random.choice([-1.0, 1.0], size=var.shape).astype(np.float32)
            v_list.append(tf.convert_to_tensor(rademacher))

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                y_pred = model(x_batch, training=False)
                loss = loss_fn(y_batch, y_pred)

            # First gradient: g = grad(loss, W)
            grads = inner_tape.gradient(loss, trainable_vars)

            # Inner product between grad and random vector: sum_i (g_i * v_i)
            gv_products = [tf.reduce_sum(g * v) for g, v in zip(grads, v_list) if g is not None]
            total_gv = tf.add_n(gv_products)

        # Second gradient: H * v = grad(total_gv, W)
        hvps = outer_tape.gradient(total_gv, trainable_vars)

        for name, v, hvp in zip(layer_names, v_list, hvps):
            if hvp is not None:
                # v^T * (H * v)
                quad_form = tf.reduce_sum(v * hvp).numpy()
                var_traces[name] += float(quad_form) / num_hvp

    # Ensure non-negative trace estimates
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
) -> Dict[str, any]:
    """Analytical Karush-Kuhn-Tucker (KKT) water-filling bit allocation under BOPs budget.

    Minimizes: sum_i [ c_i * 4^(-b_i) ]
    Subject to: sum_i [ a_i * b_i ] <= budget_bops, and b_min <= b_i <= b_max

    Stationarity gives: b_i*(lambda) = clip( 0.5 * log2( (c_i * ln 4) / (lambda * a_i) ), b_min, b_max )

    Args:
        layer_names: Names of quantized layers.
        c_coeffs: Perturbation sensitivities c_i for each layer.
        a_costs: Hardware cost weight a_i = MACs_i * b_a,i for each layer.
        budget_bops: Maximum allowed BOPs budget B.
        b_min: Minimum bitwidth limit (e.g. 2).
        b_max: Maximum bitwidth limit (e.g. 8).
        tol: Convergence tolerance for bisection.
        max_iter: Max bisection iterations.

    Returns:
        Dictionary containing optimal lambda, continuous bits, discrete integer bits, and achieved BOPs.
    """
    c = np.array(c_coeffs, dtype=np.float64)
    a = np.array(a_costs, dtype=np.float64)
    ln4 = np.log(4.0)

    # Check trivial boundary cases
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

    # Function evaluating b_i(lambda)
    def get_bits(lam: float) -> np.ndarray:
        if lam <= 0:
            return np.full_like(a, b_max)
        raw_b = 0.5 * np.log2((c * ln4) / (lam * a + 1e-12))
        return np.clip(raw_b, b_min, b_max)

    def h_budget(lam: float) -> float:
        return float(np.sum(a * get_bits(lam)) - budget_bops)

    # Bisection search on lambda
    lam_low = 1e-9
    lam_high = 1e4

    # Expand bracket if needed
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

    # Discretization: Greedy integer rounding respecting budget
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
