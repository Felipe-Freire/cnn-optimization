"""Evaluation at a fixed threshold, paired fold differences and warm-started QAT."""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pytest

from src.metrics import compute_sp_with_threshold, sp_at_threshold
from src.training import evaluate_model, paired_difference, train_fp32, train_qat

ARCH = {"input_dim": 25, "filters": 4, "kernel": 3, "dense_units": 4}


def test_sp_at_threshold_counts_accepted_events():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.7, 0.6, 0.2, 0.1, 0.3])
    sp, pd_, fa = sp_at_threshold(y, p, 0.5)
    assert pd_ == pytest.approx(0.75) and fa == pytest.approx(0.25)
    assert sp == pytest.approx(np.sqrt(np.sqrt(0.75 * 0.75) * 0.75))


def test_paired_difference():
    out = paired_difference([0.98, 0.97, 0.99], [0.97, 0.97, 0.97])
    assert out["mean"] == pytest.approx(0.01)
    assert out["std"] == pytest.approx(0.01)
    assert out["se"] == pytest.approx(0.01 / np.sqrt(3))
    assert out["n"] == 3


class _FixedScores:
    """Stand-in model returning precomputed scores."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, x, batch_size=None, verbose=0):
        return self.scores[id(x)]


def test_evaluate_model_scores_the_test_set_at_the_validation_threshold():
    rng = np.random.default_rng(0)
    y_val, y_test = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
    p_val = np.clip(0.5 * y_val + rng.normal(0.25, 0.2, 400), 0, 1)
    p_test = np.clip(0.5 * y_test + rng.normal(0.30, 0.2, 400), 0, 1)
    x_val, x_test = np.zeros((400, 1)), np.ones((400, 1))
    model = _FixedScores({id(x_val): p_val, id(x_test): p_test})

    ev = evaluate_model(model, x_val, y_val, x_test, y_test)
    threshold = compute_sp_with_threshold(y_val, p_val)[3]
    assert ev["threshold"] == pytest.approx(threshold)
    assert ev["sp_test"] == pytest.approx(sp_at_threshold(y_test, p_test, threshold)[0])
    assert ev["sp_test_knee"] >= ev["sp_test"]


def test_qat_starts_from_the_given_fp32_weights():
    rng = np.random.default_rng(1)
    x = rng.random((64, 25)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.float32)
    fp32, _, _ = train_fp32(ARCH, x, y, x, y, epochs=1, seed=3)
    qat, _, _ = train_qat(ARCH, {"conv1d_1": 8, "dense_1": 8, "output_dense": 8}, {"conv1d_1": 8, "dense_1": 8},
                          8, fp32, x, y, x, y, epochs=1, learning_rate=0.0, seed=3, reduce_lr=False)
    for name in ("conv1d_1", "dense_1", "output_dense"):
        for w_fp32, w_q in zip(fp32.get_layer(name).get_weights(), qat.get_layer(name).get_weights()):
            np.testing.assert_array_equal(w_fp32, w_q)
