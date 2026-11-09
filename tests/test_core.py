"""Unit tests for core optimization and modeling modules."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pytest
import tensorflow as tf

from src.metrics import compute_sp, norm1
from src.hardware import (
    VU13P_LIMITS,
    compute_model_hardware_costs,
    estimate_hardware_resources,
    get_vu13p_budget,
)
from src.kkt_allocator import (
    LagrangeKKTQuantizerOptimizer,
    check_kkt_stopping_criteria,
    solve_kkt_bit_allocation,
)
from src.models import build_compact_cnn1d_fp32, build_compact_qcnn1d
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


def test_models_forward_pass():
    dummy_x = np.random.randn(4, 50).astype(np.float32)

    # FP32 model
    m_fp32 = build_compact_cnn1d_fp32(input_dim=50, num_filters=4, dense_units=4)
    pred_fp32 = m_fp32(dummy_x)
    assert pred_fp32.shape == (4, 1)

    # QKeras model
    m_q = build_compact_qcnn1d(
        input_dim=50,
        num_filters=4,
        dense_units=4,
        weights_bits={"conv1d_1": 6, "dense_1": 4, "output_dense": 4},
    )
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

    # Subsequent feasible steps where multipliers decay to 0 and SP stabilizes
    for step in range(2, 7):
        sp_val = 0.9650
        sp_hist.append(sp_val)
        cost_t, g_t = opt.lagrangian_penalty(1.0 - sp_val, res_low)
        opt.update_multipliers(g_t)
        conv_t, diag_t = check_kkt_stopping_criteria(sp_hist, g_t, opt.mu, tol_sp=1e-3, tol_kkt=1e-2, window=2)
        opt.record_step(step, sp_val, {"conv1d_1": 4}, {"conv1d_1": 8}, res_low, g_t, cost_t, diag_t)

    assert conv_t
    assert diag_t["primal_feasible"]
    assert diag_t["complementary_slackness"]

    plot_path = plot_primal_dual_kkt_evolution(opt.history, save_path=str(tmp_path / "kkt_evo.png"))
    assert os.path.isfile(plot_path) and os.path.getsize(plot_path) > 0
