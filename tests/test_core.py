"""Unit tests for core optimization and modeling modules."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras

from src.hardware import (
    VU13P_CHIP_LIMITS,
    LayerSpec,
    compute_model_hardware_costs,
    csd_digits,
    get_cost_model,
    get_vu13p_budget,
    layer_features,
    layer_input_bits,
)
from src.kkt_allocator import (
    ChoiceGroup,
    LagrangianDualAllocator,
    budget_frontier,
    kkt_certificate,
    solve_exact,
    solve_kkt_bit_allocation,
)
from src.metrics import compute_sp, norm1, sp_at_threshold
from src.models import (
    build_compact_cnn1d_fp32,
    build_compact_qcnn1d,
    quantize_weights_int,
    transfer_weights_to_qmodel,
    weight_lsb,
)
from src.sensitivity import (
    compute_hutchinson_hessian_trace,
    compute_layer_c_coefficients,
    monotone_table,
)
from src.visualization import plot_pareto_frontier, plot_primal_dual_kkt_evolution, plot_sensitivity_validation


def test_norm1():
    raw_data = np.array([[2.0, 3.0, 5.0], [0.0, 0.0, 0.0]])
    normed = norm1(raw_data)
    assert np.isclose(np.sum(normed[0]), 1.0)
    assert np.all(normed[1] == 0.0)


def test_compute_sp():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    sp_val, pd_knee, fa_knee, _ = compute_sp(y_true, y_pred)
    assert pd_knee == 1.0 and fa_knee == 0.0 and np.isclose(sp_val, 1.0)


def test_sp_at_fixed_threshold():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.05])
    sp, pd_, fa = sp_at_threshold(y_true, y_pred, 0.5)
    assert pd_ == 0.75 and fa == 0.25
    assert np.isclose(sp, np.sqrt(np.sqrt(0.75 * 0.75) * 0.75))


# ----------------------------------------------------------------------------- continuous KKT


def test_water_filling_bisection_and_newton_agree_and_satisfy_kkt():
    names = ["conv1d_1", "dense_1", "output_dense"]
    c = [10.0, 2.0, 0.5]
    a = [1200.0, 2000.0, 100.0]
    budget = 16000.0
    sols = {m: solve_kkt_bit_allocation(names, c, a, budget, b_min=2, b_max=8, method=m) for m in ("bisection", "newton")}
    for sol in sols.values():
        assert sol["achieved_bops"] <= budget
        assert np.isclose(sol["continuous_cost"], budget, rtol=1e-6)        # active budget, slack = 0
        for name in names:
            assert 2 <= sol["integer_bits"][name] <= 8
            k = sol["kkt"][name]
            if k["bound"] == "interior":
                assert abs(k["stationarity_residual"]) < 1e-9
            else:
                assert k.get("nu_lower", k.get("nu_upper")) >= -1e-12     # dual feasibility of bounds
    for name in names:
        assert np.isclose(sols["bisection"]["continuous_bits"][name], sols["newton"]["continuous_bits"][name], atol=1e-6)
    assert sols["newton"]["iterations"] <= sols["bisection"]["iterations"]
    # most sensitive per unit cost (c/a) gets the most bits
    assert sols["bisection"]["continuous_bits"]["conv1d_1"] >= sols["bisection"]["continuous_bits"]["dense_1"]


def test_water_filling_inactive_budget():
    sol = solve_kkt_bit_allocation(["a", "b"], [1.0, 1.0], [1.0, 1.0], budget_bops=100.0, b_min=2, b_max=8)
    assert not sol["budget_active"] and sol["lambda_optimal"] == 0.0
    assert sol["integer_bits"] == {"a": 8, "b": 8}


# ----------------------------------------------------------------------------- sensitivity


def test_c_coefficients_have_no_n_factor():
    """Two layers with the same Hessian trace get the same c_l regardless of their size (qkeras range)."""
    m = build_compact_cnn1d_fp32(input_dim=25, num_filters=8, kernel_size=3, dense_units=8)
    c = compute_layer_c_coefficients(m, {"conv1d_1": 1.0, "dense_1": 1.0, "output_dense": 1.0})
    assert np.isclose(c["conv1d_1"], c["dense_1"]) and np.isclose(c["dense_1"], c["output_dense"])
    assert np.isclose(c["conv1d_1"], 8.0 ** 2 / 24.0)


def test_hutchinson_trace_matches_analytic_hessian():
    """Linear model + MSE: H = 2/N X^T X, Tr(H) = 2/N sum ||x||^2."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 6)).astype(np.float32)
    y = rng.normal(size=(256, 1)).astype(np.float32)
    model = keras.Sequential([keras.layers.Input(shape=(6,)), keras.layers.Dense(1, name="lin")])
    tr = compute_hutchinson_hessian_trace(model, keras.losses.MeanSquaredError(), tf.constant(X), tf.constant(y), num_hvp=200)
    assert np.isclose(tr["lin"], 2.0 / len(X) * np.sum(X ** 2), rtol=0.1)


def test_monotone_table():
    t = monotone_table({"l": {2: 0.5, 4: 0.01, 6: 0.02, 8: -1e-4}})["l"]
    assert t[8] == 0.0 and t[6] == 0.02 and t[4] == 0.02 and t[2] == 0.5


# ----------------------------------------------------------------------------- hardware model


def test_csd_digits():
    assert [csd_digits(n) for n in (0, 1, 8, 7, 5, 11, -3, -16)] == [0, 1, 1, 2, 2, 3, 2, 1]


def test_layer_features_zero_and_power_of_two_weights_are_free():
    zeros = LayerSpec("z", np.zeros((4, 2), int), np.zeros(2, int), w_bits=8, a_in_bits=8)
    assert layer_features(zeros)["adders"] == 0.0
    pow2 = LayerSpec("p", np.array([[1, 2], [4, -8], [16, 32]]), np.zeros(2, int), w_bits=8, a_in_bits=8)
    f = layer_features(pow2)
    assert f["adders"] == 2 * (3 - 1) and f["DSP"] == 0.0     # only the adder tree
    wide = LayerSpec("w", np.array([[11], [7]]), np.zeros(1, int), w_bits=16, a_in_bits=16)
    assert layer_features(wide)["DSP"] == 1.0                  # 11 (3 CSD digits) -> DSP, 7 (2 digits) -> shift-add


def test_cost_model_reproduces_synthesis_points():
    cm = get_cost_model()
    table = pd.DataFrame(cm.calibration_table())
    dom = table[table["in_domain"]]
    assert len(dom) >= 5
    assert ((dom["DSP_predicted"] - dom["DSP_measured"]).abs() <= np.maximum(0.05 * dom["DSP_measured"], 2)).all()
    assert (dom["LUT_rel_err"].abs() < 0.25).all() and (dom["FF_rel_err"].abs() < 0.25).all()
    assert dom["LUT_loo_rel_err"].notna().all()                    # leave-one-out available with >= 4 points
    m = cm.margins
    assert all(0.0 <= v < 0.5 for v in m.values())
    assert np.isclose(m["LUT"], max(0.0, -dom["LUT_loo_rel_err"].min()))
    assert not table[~table["in_domain"]].empty  # PF=1 point kept as out-of-domain evidence


def test_relu_bits_feature():
    k = np.ones((3, 2), int)
    on = layer_features(LayerSpec("c", k, np.zeros(2, int), w_bits=4, a_in_bits=8, positions=5, relu_after=True))
    off = layer_features(LayerSpec("c", k, np.zeros(2, int), w_bits=4, a_in_bits=8, positions=5))
    assert on["relu_bits"] == 5 * 2 * (8 + 4 + 2) and off["relu_bits"] == 0.0
    assert on["adder_bits"] == off["adder_bits"]


def test_activation_mapping_uses_input_precision_of_each_layer():
    assert layer_input_bits(["conv1d_1", "dense_1", "output_dense"], {"conv1d_1": 8, "dense_1": 6}, input_bits=10) == {
        "conv1d_1": 10, "dense_1": 8, "output_dense": 6}
    m_q = build_compact_qcnn1d(input_dim=25, num_filters=8, dense_units=8,
                               weights_bits={"conv1d_1": 4, "dense_1": 6, "output_dense": 8},
                               act_bits={"conv1d_1": 8, "dense_1": 6}, input_bits=8)
    m_q(np.zeros((1, 25), np.float32))
    costs = compute_model_hardware_costs(m_q, weights_bits={"conv1d_1": 4, "dense_1": 6, "output_dense": 8},
                                         act_bits={"conv1d_1": 8, "dense_1": 6}, input_bits=8)
    assert [b["b_a_in"] for b in costs["breakdown"]] == [8, 8, 6]
    assert costs["total_macs"] == 600 + 768 + 8
    assert costs["total_bops"] == 600 * 4 * 8 + 768 * 6 * 8 + 8 * 8 * 6
    assert costs["vu13p_resources"]["BRAM_18K"] == 1.0


def test_quantize_weights_int_matches_qkeras_grid():
    w = np.array([0.1, -0.3, 0.26, 3.9, -5.0], dtype=np.float32)
    k = quantize_weights_int(w, 5)
    assert np.allclose(k * weight_lsb(5), [0.0, -0.25, 0.25, 3.75, -4.0])


def test_vu13p_budget():
    b = get_vu13p_budget(n_instances=100, device_fraction=0.6)
    assert np.isclose(b["LUT"], 0.6 * VU13P_CHIP_LIMITS["LUT"] / 100)
    assert set(b) == {"DSP", "LUT", "FF"}


def test_models_forward_pass_and_warm_start():
    x = np.random.rand(4, 25).astype(np.float32)
    m_fp32 = build_compact_cnn1d_fp32(input_dim=25, num_filters=4, dense_units=4)
    m_q = build_compact_qcnn1d(input_dim=25, num_filters=4, dense_units=4,
                               weights_bits={"conv1d_1": 6, "dense_1": 4, "output_dense": 4}, input_bits=8)
    transfer_weights_to_qmodel(m_fp32, m_q)
    assert m_q(x).shape == (4, 1)
    assert np.allclose(m_q.get_layer("dense_1").get_weights()[0], m_fp32.get_layer("dense_1").get_weights()[0])


# ----------------------------------------------------------------------------- discrete solvers


def _toy_groups(seed=0, n_groups=3, n_opt=6):
    rng = np.random.default_rng(seed)
    groups = []
    for g in range(n_groups):
        bits = np.arange(2, 2 + n_opt)
        omega = np.sort(rng.uniform(0.1, 1.0, n_opt))[::-1] * np.exp(-0.6 * (bits - 2))
        lut = bits * rng.uniform(50, 150) + rng.uniform(0, 30, n_opt)
        ff = 0.4 * lut
        dsp = np.where(bits > 6, rng.integers(1, 5), 0).astype(float)
        options = [{"w": {f"l{g}": int(b)}, "a": {}} for b in bits]
        groups.append(ChoiceGroup(f"l{g}", options, omega, {"LUT": lut, "FF": ff, "DSP": dsp}))
    return groups


def test_dual_decomposition_brackets_and_finds_exact_optimum():
    groups = _toy_groups()
    budgets = {"LUT": 1500.0, "FF": 2000.0, "DSP": 3.0}
    overhead = {"LUT": 0.0, "FF": 0.0, "DSP": 0.0}
    exact = solve_exact(groups, budgets, overhead)
    assert exact["feasible"] and all(v <= 1e-12 for v in exact["g"].values())
    dual = LagrangianDualAllocator(groups, budgets, overhead).solve(max_iter=300)
    assert dual["lower_bound"] <= exact["omega"] + 1e-12 <= dual["upper_bound"] + 2e-12
    assert all(v <= 1e-12 for v in dual["g"].values())
    assert np.isclose(dual["omega"], exact["omega"])
    assert all(m >= 0 for m in dual["mu"].values())


def test_dual_without_heuristic_is_still_a_valid_bound():
    groups = _toy_groups(seed=3)
    budgets = {"LUT": 1400.0, "FF": 5000.0, "DSP": 10.0}
    overhead = {"LUT": 0.0, "FF": 0.0, "DSP": 0.0}
    exact = solve_exact(groups, budgets, overhead)
    dual = LagrangianDualAllocator(groups, budgets, overhead, use_heuristic=False).solve(max_iter=300)
    assert dual["lower_bound"] <= exact["omega"] + 1e-12 <= dual["upper_bound"] + 1e-12


def test_kkt_certificate_inactive_constraints():
    cert = kkt_certificate(omega_x=0.1, dual_value=0.1, mu={"LUT": 0.0, "FF": 0.0}, g={"LUT": -0.2, "FF": -0.5})
    assert cert["certified_optimal"] and cert["duality_gap"] == 0.0
    bad = kkt_certificate(omega_x=0.1, dual_value=0.05, mu={"LUT": 0.3}, g={"LUT": -0.2})
    assert not bad["complementary_slackness_ok"] and not bad["certified_optimal"]


def test_budget_frontier_is_monotone():
    groups = _toy_groups(seed=1)
    budgets = {"LUT": 1000.0, "FF": 1e9, "DSP": 1e9}
    overhead = {"LUT": 0.0, "FF": 0.0, "DSP": 0.0}
    front = [s for s in budget_frontier(groups, budgets, overhead, "LUT", [600, 900, 1200, 1500, 2000]) if s["feasible"]]
    omegas = [s["omega"] for s in front]
    assert omegas == sorted(omegas, reverse=True)


# ----------------------------------------------------------------------------- plots


def test_new_plots(tmp_path):
    groups = _toy_groups()
    alloc = LagrangianDualAllocator(groups, {"LUT": 1500.0, "FF": 2000.0, "DSP": 3.0}, {})
    alloc.solve(max_iter=30)
    p1 = plot_primal_dual_kkt_evolution(alloc.history, save_path=str(tmp_path / "kkt.png"), exact_optimum=0.5)
    om = {"conv1d_1": {2: 0.1, 4: 0.01, 8: 1e-4}}
    p2 = plot_sensitivity_validation(om, om, save_path=str(tmp_path / "sens.png"))
    front = pd.DataFrame({"objective": ["hessian", "hessian"], "LUT": [5000.0, 9000.0], "sp_val": [0.97, 0.975]})
    uni = pd.DataFrame({"bits": [4, 8], "LUT": [6000.0, 24000.0], "sp_val": [0.972, 0.978]})
    p3 = plot_pareto_frontier(front, uni, lut_budget=10000.0, save_path=str(tmp_path / "pareto.png"), fp32_sp=0.978)
    for p in (p1, p2, p3):
        assert os.path.isfile(p) and os.path.getsize(p) > 0


def test_layer_specs_mark_relu_layers_in_fp32_and_qkeras_models():
    m_fp = build_compact_cnn1d_fp32(input_dim=25, num_filters=8, kernel_size=3, dense_units=8)
    m_q = build_compact_qcnn1d(input_dim=25, num_filters=8, dense_units=8, input_bits=8)
    m_q(np.zeros((1, 25), np.float32))
    from src.hardware import layer_specs_from_model
    for m in (m_fp, m_q):
        flags = {s.name: s.relu_after for s in layer_specs_from_model(m, input_bits=8)}
        assert flags == {"conv1d_1": True, "dense_1": True, "output_dense": False}
