"""Unit tests for core optimization and modeling modules."""

import itertools
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pytest
import tensorflow as tf

from src.metrics import compute_sp, norm1
from src.hardware import (
    RAMB18_BITS,
    VU13P_LIMITS,
    LayerSpec,
    calibrated_resources,
    compute_model_hardware_costs,
    csd_digits,
    estimate_hardware_resources,
    get_cost_model,
    get_vu13p_budget,
    layer_features,
    layer_input_bits,
    make_cost_fn,
)
from src.kkt_allocator import (
    LagrangeKKTQuantizerOptimizer,
    check_kkt_stopping_criteria,
    compute_hutchinson_hessian_trace,
    compute_layer_c_coefficients,
    solve_kkt_bit_allocation,
)
from src.models import build_compact_cnn1d_fp32, build_compact_qcnn1d, transfer_weights_to_qmodel
from src.visualization import plot_primal_dual_kkt_evolution


def test_norm1():
    # Test positive energy rings typical of calorimeter deposits
    raw_data = np.array([
        [2.0, 3.0, 5.0],
        [0.0, 0.0, 0.0],
    ])
    normed = norm1(raw_data)
    assert np.isclose(np.sum(normed[0]), 1.0)
    assert np.all(normed[1] == 0.0)


def test_compute_sp():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    sp_val, pd_knee, fa_knee, _ = compute_sp(y_true, y_pred)
    assert 0.0 <= sp_val <= 1.0
    assert pd_knee == 1.0
    assert fa_knee == 0.0
    assert np.isclose(sp_val, 1.0)


def test_kkt_bit_allocation():
    layer_names = ["conv1d_1", "dense_1", "output_dense"]
    # Layer 1 is sensitive but moderate cost; Layer 2 has high cost; Layer 3 is small
    c_coeffs = [10.0, 2.0, 0.5]
    a_costs = [1200.0, 2000.0, 100.0]  # MACs * b_a
    budget = 16000.0  # Allowed BOPs

    res = solve_kkt_bit_allocation(
        layer_names, c_coeffs, a_costs, budget_bops=budget, b_min=2, b_max=8
    )

    int_bits = res["integer_bits"]
    achieved_bops = res["achieved_bops"]

    assert achieved_bops <= budget
    for name in layer_names:
        assert 2 <= int_bits[name] <= 8

    # The most sensitive layer should get >= bits than less sensitive high-cost layer
    assert int_bits["conv1d_1"] >= int_bits["dense_1"]


def _small_model_and_batch(seed: int = 0):
    tf.random.set_seed(seed)
    model = build_compact_cnn1d_fp32(input_dim=25, num_filters=4, kernel_size=3, dense_units=4)
    rng = np.random.default_rng(seed)
    x = rng.random((256, 25)).astype(np.float32)
    x /= x.sum(axis=1, keepdims=True)
    y = (x[:, :12].sum(axis=1) > 0.48).astype(np.float32)[:, None]
    return model, tf.constant(x), tf.constant(y)


def _exact_hessian_block(model, loss_fn, x, y, layer):
    var = layer.kernel
    n = int(np.prod(var.shape))
    with tf.GradientTape(persistent=True) as outer:
        with tf.GradientTape() as inner:
            loss = loss_fn(y, model(x, training=False))
        grad = tf.reshape(inner.gradient(loss, var), [-1])
    jac = outer.jacobian(grad, var, experimental_use_pfor=False)
    return tf.reshape(jac, [n, n]).numpy().astype(np.float64)


def test_hutchinson_probes_each_layer_block_alone():
    """The estimate is the mean of z^T H_ll z over the draws of that layer, with no cross-layer terms."""
    model, x, y = _small_model_and_batch()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    num_hvp, seed = 8, 3
    numpy_state = np.random.get_state()[1].copy()

    traces = compute_hutchinson_hessian_trace(model, loss_fn, x, y, num_hvp=num_hvp, seed=seed)

    assert np.array_equal(np.random.get_state()[1], numpy_state)  # global RNG untouched
    rng = np.random.default_rng(seed)  # same draw order: layer by layer, num_hvp draws each
    for layer in [l for l in model.layers if getattr(l, "kernel", None) is not None]:
        h = _exact_hessian_block(model, loss_fn, x, y, layer)
        draws = [rng.choice([-1.0, 1.0], size=layer.kernel.shape).ravel() for _ in range(num_hvp)]
        expected = float(np.mean([z @ h @ z for z in draws]))
        assert traces[layer.name] == pytest.approx(max(expected, 1e-8), rel=1e-3, abs=1e-9)


def test_layer_c_coefficients_do_not_depend_on_layer_size_or_weight_range():
    """c_l = Tr(H_ll) kappa^2 / 24: the full trace already sums over the n_l weights, and the
    trained quantizer has a fixed range [-4, 4) (kappa = 8), whatever max|W_l| is."""
    model = build_compact_cnn1d_fp32(input_dim=25, num_filters=8, kernel_size=3, dense_units=8)
    names = ["conv1d_1", "dense_1", "output_dense"]
    c = compute_layer_c_coefficients(model, {name: 1.0 for name in names})
    for name in names:
        assert c[name] == pytest.approx(64.0 / 24.0)

    dense = model.get_layer("dense_1")
    kernel, bias = dense.get_weights()
    dense.set_weights([10.0 * kernel, bias])
    assert compute_layer_c_coefficients(model, {"dense_1": 3.0})["dense_1"] == pytest.approx(3.0 * 64.0 / 24.0)


LAYERS = ["conv1d_1", "dense_1", "output_dense"]
MACS = [600.0, 768.0, 8.0]
PARAMS = [32.0, 776.0, 9.0]


def test_layer_input_bits_follow_the_data_path():
    a_in = layer_input_bits(LAYERS, {"conv1d_1": 5, "dense_1": 3}, input_bits=7)
    assert a_in == {"conv1d_1": 7, "dense_1": 5, "output_dense": 3}


@pytest.mark.parametrize(
    "act_bits, expected_a_in",
    [({"conv1d_1": 8, "dense_1": 4}, [8, 8, 4]), ({"conv1d_1": 4, "dense_1": 8}, [8, 4, 8])],
)
def test_hardware_costs_use_the_precision_entering_each_layer(act_bits, expected_a_in):
    """dense_1 multiplies the ReLU of conv1d_1 and output_dense the ReLU of dense_1; QTools agrees."""
    w_bits = {name: 8 for name in LAYERS}
    model = build_compact_qcnn1d(input_dim=25, weights_bits=w_bits, act_bits=act_bits, input_bits=8)
    hw = compute_model_hardware_costs(model, weights_bits=w_bits, act_bits=act_bits, input_dim=25, input_bits=8)
    qtools_layers = hw["qtools_vu13p_detailed"]["qtools_profile"]["layers"]

    assert [row["b_a"] for row in hw["breakdown"]] == expected_a_in
    assert [qtools_layers[name]["input_bits"] for name in LAYERS] == expected_a_in
    assert hw["total_bops"] == sum(m * 8 * a for m, a in zip(MACS, expected_a_in))


def test_input_quantizer_matches_the_costed_input_format():
    """input_bits=b quantizes the (non-negative) rings to unsigned ufixed<b, 0>: step 2^-b in [0, 1)."""
    model = build_compact_qcnn1d(input_dim=25, input_bits=6)
    x = np.linspace(-0.2, 1.2, 25, dtype=np.float32)[None, :]
    xq = model.get_layer("input_quant")(x).numpy()
    step = 2.0 ** -6
    assert np.allclose(xq / step, np.round(xq / step))
    assert xq.min() == 0.0 and xq.max() <= 1.0 - step + 1e-7

    hw = compute_model_hardware_costs(model, input_dim=25, input_bits=6)
    assert hw["breakdown"][0]["b_a"] == 6


def test_primal_update_costs_weight_bits_with_the_input_precision():
    """A 2-bit ReLU after dense_1 only makes output_dense cheaper: the allocation must fit the LUT
    budget when each layer is costed with the precision that enters it."""
    act_bits = {"conv1d_1": 8, "dense_1": 2}
    opt = LagrangeKKTQuantizerOptimizer({"DSP": 0.0, "LUT": 2500.0, "FF": 10000.0, "BRAM_bits": 5500.0})
    w, _, _ = opt.compute_primal_bit_update(
        LAYERS, [0.4, 1.3, 0.01], MACS, PARAMS, {name: 8 for name in LAYERS}, act_bits
    )
    a_in = layer_input_bits(LAYERS, act_bits, input_bits=8)
    used = estimate_hardware_resources([
        {"w_bits": w[name], "a_bits": a_in[name], "params": p, "macc": m}
        for name, m, p in zip(LAYERS, MACS, PARAMS)
    ])
    assert used["LUT"] <= 2500.0


BUDGET_LUT_BINDING = {"DSP": 0.0, "LUT": 2500.0, "FF": 10000.0, "BRAM_bits": 5500.0}
C_TEST = [0.4, 1.3, 0.01]


def _omega(c, w):
    return float(np.sum(np.asarray(c) * 4.0 ** (-np.asarray(w, dtype=float))))


def _resources(w_bits, a_in_bits=(8, 8, 8)):
    return estimate_hardware_resources([
        {"w_bits": w, "a_bits": a, "params": p, "macc": m}
        for w, a, p, m in zip(w_bits, a_in_bits, PARAMS, MACS)
    ])


def test_primal_step_matches_water_filling_when_only_lut_binds():
    """With one binding resource the prices reduce to the single-budget water-filling:
    same continuous bits, and mu_LUT = lambda * 18 * B_LUT; resources with slack get price 0."""
    opt = LagrangeKKTQuantizerOptimizer(dict(BUDGET_LUT_BINDING))
    _, _, sol = opt.compute_primal_bit_update(
        LAYERS, C_TEST, MACS, PARAMS, {n: 8 for n in LAYERS}, {"conv1d_1": 8, "dense_1": 8}
    )
    ref = solve_kkt_bit_allocation(LAYERS, C_TEST, [m * 8 for m in MACS], budget_bops=2500.0 * 18)

    for name in LAYERS:
        assert sol["continuous_bits"][name] == pytest.approx(ref["continuous_bits"][name], abs=1e-4)
    assert opt.mu["LUT"] == pytest.approx(ref["lambda_optimal"] * 18 * 2500.0, rel=1e-3)
    assert opt.mu["FF"] == pytest.approx(0.0, abs=1e-9)
    assert opt.mu["BRAM_bits"] == pytest.approx(0.0, abs=1e-9)
    assert sol["g_continuous"]["LUT"] == pytest.approx(0.0, abs=1e-5)


def test_primal_step_satisfies_kkt_with_two_binding_resources():
    """LUT and BRAM both bind: stationarity, feasibility, complementary slackness and weak duality."""
    opt = LagrangeKKTQuantizerOptimizer({**BUDGET_LUT_BINDING, "BRAM_bits": 3200.0})
    _, _, sol = opt.compute_primal_bit_update(
        LAYERS, C_TEST, MACS, PARAMS, {n: 8 for n in LAYERS}, {"conv1d_1": 8, "dense_1": 8}
    )
    assert sol["dual_converged"]
    assert sol["stationarity_residual"] < 1e-9
    assert max(sol["g_continuous"].values()) < 1e-5
    assert sol["complementary_slackness"] < 1e-5
    assert opt.mu["LUT"] > 0.0 and opt.mu["BRAM_bits"] > 0.0
    assert sol["omega_integer"] >= sol["dual_bound"] - 1e-12


def test_primal_step_returns_the_integer_optimum():
    """The integer bits are the best feasible combination under the cost model (BRAM binding)."""
    budget = {**BUDGET_LUT_BINDING, "BRAM_bits": 3200.0}
    opt = LagrangeKKTQuantizerOptimizer(budget)
    w, _, _ = opt.compute_primal_bit_update(
        LAYERS, C_TEST, MACS, PARAMS, {n: 8 for n in LAYERS}, {"conv1d_1": 8, "dense_1": 8}
    )
    feasible = [
        cand for cand in itertools.product(range(2, 9), repeat=3)
        if all(_resources(cand)[k] <= budget[k] + 1e-9 for k in budget)
    ]
    chosen = tuple(w[n] for n in LAYERS)
    assert chosen in feasible
    assert chosen == min(feasible, key=lambda cand: _omega(C_TEST, cand))


def test_stopping_rule_rejects_a_configuration_without_stationarity():
    """A frozen configuration (2 bits everywhere, SP = 0.5) that did not come from the Lagrangian
    minimization is never certified, even after its prices settle."""
    res = _resources((2, 2, 2), a_in_bits=(8, 2, 2))
    opt = LagrangeKKTQuantizerOptimizer(dict(BUDGET_LUT_BINDING), lr_mu=0.8)
    opt.mu = {"DSP": 4.6, "LUT": 0.766, "FF": 0.089, "BRAM_bits": 1.252}
    sp_history = []
    for _ in range(30):
        sp_history.append(0.5)
        _, g = opt.lagrangian_penalty(0.5, res)
        opt.update_multipliers(g)
        converged, diag = check_kkt_stopping_criteria(sp_history, g, opt.mu, tol_sp=1e-3, tol_kkt=2e-2, window=2)
        assert not converged
    assert not diag["stationary"]


def test_lagrangian_counts_slack_constraints():
    """L(x, mu) = (1 - SP) + sum_j mu_j g_j: a priced constraint with slack lowers L."""
    opt = LagrangeKKTQuantizerOptimizer({"LUT": 1000.0})
    opt.mu = {"LUT": 0.5}
    cost, g = opt.lagrangian_penalty(0.1, {"LUT": 800.0})
    assert g["LUT"] == pytest.approx(-0.2)
    assert cost == pytest.approx(0.1 - 0.5 * 0.2)


def test_csd_digits():
    expected = {0: 0, 1: 1, 2: 1, -4: 1, 3: 2, 7: 2, -7: 2, 5: 2, 23: 3, 255: 2}
    assert {n: csd_digits(n) for n in expected} == expected


def test_zero_and_power_of_two_weights_need_no_extra_adders():
    """Each nonzero term adds to the neuron's adder tree; only non-power-of-two constants add more."""
    shifts = LayerSpec("d", np.array([[0], [1], [2], [-4]]), np.array([0]), w_bits=4, a_in_bits=8)
    assert layer_features(shifts)["adders"] == 2.0  # 3 nonzero terms -> 2 adders, no multipliers
    seven = LayerSpec("d", np.array([[7], [0], [0], [0]]), np.array([0]), w_bits=4, a_in_bits=8)
    assert layer_features(seven)["adders"] == 1.0  # 7 = 8 - 1 -> one adder
    assert layer_features(seven)["DSP"] == 0.0


def test_cost_model_reproduces_the_synthesized_points():
    """Fit on the hls4ml/Vivado points of this network; in-domain errors stay within the observed range."""
    cm = get_cost_model()
    rows = [r for r in cm.calibration_table() if r["in_domain"]]
    assert len(rows) >= 6
    for r in rows:
        assert abs(r["LUT_rel_err"]) < 0.20, r
        assert abs(r["FF_rel_err"]) < 0.15, r
        assert abs(r["DSP_predicted"] - r["DSP_measured"]) <= max(2.0, 0.01 * r["DSP_measured"]), r
    assert all(0.0 < m < 0.5 for m in cm.margins.values())


def test_budget_fraction_and_margins():
    margins = {"LUT": 0.125, "DSP": 0.2}
    budget = get_vu13p_budget({"DSP": 0.01, "LUT": 0.01, "FF": 0.01, "BRAM_bits": 0.01}, margins=margins)
    assert budget["LUT"] == pytest.approx(0.01 * VU13P_LIMITS["LUT"] / 1.125)
    assert budget["DSP"] == pytest.approx(0.01 * VU13P_LIMITS["DSP"] / 1.2)
    assert budget["FF"] == pytest.approx(0.01 * VU13P_LIMITS["FF"])


def _seeded_fp32_model(seed: int = 0):
    import tf_keras
    tf_keras.utils.set_random_seed(seed)
    return build_compact_cnn1d_fp32(input_dim=25, num_filters=8, kernel_size=3, dense_units=8)


def test_calibrated_costs_of_a_model():
    """Weights are constants in the logic: one RAMB18 (sigmoid), no DSP up to 8 bits, fewer LUTs
    with fewer weight bits, and the per-layer costs add up to the total."""
    model = _seeded_fp32_model()
    w8 = {name: 8 for name in LAYERS}
    w4 = {name: 4 for name in LAYERS}
    hw8 = compute_model_hardware_costs(model, weights_bits=w8, act_bits=8, input_dim=25)
    hw4 = compute_model_hardware_costs(model, weights_bits=w4, act_bits=8, input_dim=25)

    assert hw8["vu13p_resources"]["BRAM_bits"] == RAMB18_BITS
    assert hw8["vu13p_resources"]["DSP"] == 0.0
    assert hw4["vu13p_resources"]["LUT"] < hw8["vu13p_resources"]["LUT"]
    assert sum(row["LUT"] for row in hw8["breakdown"]) == pytest.approx(hw8["vu13p_resources"]["LUT"])
    assert sum(row["FF"] for row in hw8["breakdown"]) + get_cost_model().gamma_ff == pytest.approx(
        hw8["vu13p_resources"]["FF"]
    )


def test_cost_fn_matches_calibrated_resources():
    model = _seeded_fp32_model()
    cost_fn = make_cost_fn(model, input_bits=8)
    for w_bits, act_bits in [
        ({"conv1d_1": 3, "dense_1": 5, "output_dense": 8}, {"conv1d_1": 8, "dense_1": 8}),
        ({"conv1d_1": 6, "dense_1": 4, "output_dense": 12}, {"conv1d_1": 6, "dense_1": 12}),
    ]:
        expected = calibrated_resources(model, w_bits, act_bits, input_bits=8)["total"]
        got = cost_fn(w_bits, act_bits)
        for key in ("DSP", "LUT", "FF", "BRAM_bits"):
            assert got[key] == pytest.approx(expected[key])


def test_allocator_returns_the_integer_optimum_of_the_calibrated_model():
    """With the calibrated cost function the allocation is the best feasible combination of that model."""
    model = _seeded_fp32_model()
    cost_fn = make_cost_fn(model, input_bits=8)
    act = {"conv1d_1": 8, "dense_1": 8}
    lut_low = cost_fn({n: 2 for n in LAYERS}, act)["LUT"]
    lut_high = cost_fn({n: 8 for n in LAYERS}, act)["LUT"]
    budget = {"DSP": 102.0, "LUT": 0.5 * (lut_low + lut_high), "FF": 28000.0, "BRAM_bits": 990904.0}

    opt = LagrangeKKTQuantizerOptimizer(budget)
    w, _, sol = opt.compute_primal_bit_update(
        LAYERS, C_TEST, MACS, PARAMS, {n: 8 for n in LAYERS}, act, cost_fn=cost_fn
    )
    feasible = [
        cand for cand in itertools.product(range(2, 9), repeat=3)
        if all(cost_fn(dict(zip(LAYERS, cand)), act)[k] <= budget[k] + 1e-9 for k in budget)
    ]
    chosen = tuple(w[n] for n in LAYERS)
    assert chosen == min(feasible, key=lambda cand: _omega(C_TEST, cand))
    assert opt.mu["LUT"] > 0.0
    assert sol["integer_resources"]["LUT"] <= budget["LUT"]


def test_models_forward_pass():
    dummy_x = np.random.randn(4, 50).astype(np.float32)

    # FP32 model
    m_fp32 = build_compact_cnn1d_fp32(input_dim=50, num_filters=4, dense_units=4)
    pred_fp32 = m_fp32(dummy_x)
    assert pred_fp32.shape == (4, 1)

    # QKeras model + Warm-Start weight transfer
    m_q = build_compact_qcnn1d(
        input_dim=50,
        num_filters=4,
        dense_units=4,
        weights_bits={"conv1d_1": 6, "dense_1": 4, "output_dense": 4},
    )
    transfer_weights_to_qmodel(m_fp32, m_q)
    pred_q = m_q(dummy_x)
    assert pred_q.shape == (4, 1)

    costs = compute_model_hardware_costs(
        m_q,
        weights_bits={"conv1d_1": 6, "dense_1": 4, "output_dense": 4},
        act_bits={"conv1d_1": 8, "dense_1": 6},
        input_dim=50,
    )
    assert costs["total_macs"] > 0
    assert costs["total_bops"] > 0
    assert "vu13p_resources" in costs
    assert len(costs["breakdown"]) == 3


def test_vu13p_estimator_and_primal_dual_kkt(tmp_path):
    """Verify VU13P resource estimation, LagrangeKKTQuantizerOptimizer, and KKT stopping criteria."""
    assert VU13P_LIMITS["DSP"] == 12288.0
    assert VU13P_LIMITS["LUT"] == 1728000.0
    assert VU13P_LIMITS["FF"] == 3456000.0

    # Low bitwidth (<=8b) -> LUTs, 0 DSPs
    res_low = estimate_hardware_resources([
        {"w_bits": 4, "a_bits": 8, "params": 100, "macc": 1000}
    ])
    assert res_low["DSP"] == 0.0
    assert res_low["LUT"] > 0.0

    # High bitwidth (16b) -> allocates DSPs
    res_high = estimate_hardware_resources([
        {"w_bits": 16, "a_bits": 16, "params": 100, "macc": 1000}
    ])
    assert res_high["DSP"] > 0.0

    # Test Primal-Dual Optimizer & KKT stopping criteria
    limits = get_vu13p_budget({"DSP": 0.0, "LUT": 3000.0, "FF": 8000.0, "BRAM_bits": 6000.0})
    opt = LagrangeKKTQuantizerOptimizer(hardware_limits=limits, lr_mu=0.5, normalize_constraints=True)

    # Step 1: Violates DSP and LUT
    sp_hist = [0.960]
    cost1, g1 = opt.lagrangian_penalty(1.0 - 0.960, res_high)
    opt.update_multipliers(g1)
    assert opt.mu["DSP"] > 0.0
    conv1, diag1 = check_kkt_stopping_criteria(sp_hist, g1, opt.mu)
    assert not conv1
    assert not diag1["primal_feasible"]
    opt.record_step(1, 0.960, {"conv1d_1": 16}, {"conv1d_1": 16}, res_high, g1, cost1, diag1)

    # Projected subgradient: a constraint with slack lowers its price by lr_mu * g, never below 0
    mu_before = dict(opt.mu)
    _, g_low = opt.lagrangian_penalty(1.0 - 0.965, res_low)
    opt.update_multipliers(g_low)
    for k in opt.mu:
        assert opt.mu[k] == pytest.approx(max(0.0, mu_before[k] + 0.5 * g_low[k]))

    # Closed loop: every configuration comes from compute_primal_bit_update, which reports the
    # stationarity residual the stopping rule requires
    c_coeffs = [0.05, 1.5, 0.2]
    w_cfg, a_cfg, sol = opt.compute_primal_bit_update(
        LAYERS, c_coeffs, MACS, PARAMS, {n: 16 for n in LAYERS}, {"conv1d_1": 16, "dense_1": 16}
    )
    assert all(2 <= v <= 8 for v in w_cfg.values()) and all(v <= 8 for v in a_cfg.values())
    assert w_cfg["dense_1"] >= w_cfg["conv1d_1"]

    for step in range(2, 8):
        sp_hist.append(0.965)
        a_in = layer_input_bits(LAYERS, a_cfg, input_bits=8)
        res_t = estimate_hardware_resources([
            {"w_bits": w_cfg[n], "a_bits": a_in[n], "params": p, "macc": m}
            for n, m, p in zip(LAYERS, MACS, PARAMS)
        ])
        cost_t, g_t = opt.lagrangian_penalty(1.0 - 0.965, res_t)
        opt.update_multipliers(g_t)
        conv_t, diag_t = check_kkt_stopping_criteria(
            sp_hist, g_t, opt.mu, tol_sp=1e-3, tol_kkt=1e-2, window=2,
            stationarity_residual=sol["stationarity_residual"],
        )
        opt.record_step(step, 0.965, w_cfg, a_cfg, res_t, g_t, cost_t, diag_t)
        if conv_t:
            break
        w_cfg, a_cfg, sol = opt.compute_primal_bit_update(LAYERS, c_coeffs, MACS, PARAMS, w_cfg, a_cfg)

    assert conv_t
    assert diag_t["stationary"] and diag_t["primal_feasible"] and diag_t["complementary_slackness"]

    plot_path = plot_primal_dual_kkt_evolution(opt.history, save_path=str(tmp_path / "kkt_evo.png"))
    assert os.path.isfile(plot_path) and os.path.getsize(plot_path) > 0
