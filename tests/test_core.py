"""Unit tests for core optimization and modeling modules."""

import numpy as np
import pytest
import tensorflow as tf

from src.metrics import compute_sp, norm1
from src.hardware import compute_model_hardware_costs
from src.kkt_allocator import solve_kkt_bit_allocation
from src.models import build_compact_cnn1d_fp32, build_compact_qcnn1d


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

    costs = compute_model_hardware_costs(m_q, input_dim=50)
    assert costs["total_macs"] > 0
    assert costs["total_bops"] > 0
    assert len(costs["breakdown"]) == 3
