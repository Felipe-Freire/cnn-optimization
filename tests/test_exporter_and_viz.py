"""Unit tests for Jubiaba hierarchical exporter, ring slicing (v0/v1/v2), and visualizations."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import json
import numpy as np
import pandas as pd
import pytest
import tf_keras as keras

from src.exporter import ArtifactExporter
from src.metrics import CONFIG_VERSOES, ESTRUTURA_BASE, compute_sp_with_threshold, slice_rings_by_version
from src.models import build_compact_cnn1d_fp32
from src.visualization import (
    plot_best_model_roc,
    plot_confusion_matrix_knee,
    plot_cv_roc_envelope,
    plot_cv_statistics_boxplots,
    plot_fa_per_fold_workpoints,
    plot_learning_curves,
    plot_output_score_histogram,
    plot_rings_profile,
    plot_roc_duel_fp32_vs_kkt,
    plot_vu13p_resource_utilization,
)


def test_slice_rings_by_version():
    """Verify ring slicing for v0 (100), v1 (50), and v2 (25) topologies."""
    rng = np.random.default_rng(42)
    mat_v0 = rng.normal(size=(20, 100)).astype(np.float32)

    # v0: 100 rings
    sliced_0, struct_0 = slice_rings_by_version(mat_v0, 0)
    assert sliced_0.shape == (20, 100)
    assert sum(struct_0.values()) == 100
    assert struct_0 == ESTRUTURA_BASE

    # v1: 50 rings
    sliced_1, struct_1 = slice_rings_by_version(mat_v0, 1)
    assert sliced_1.shape == (20, 50)
    assert sum(struct_1.values()) == 50
    assert struct_1["PS"] == 4
    assert struct_1["EM1"] == 32
    assert struct_1["HAD3"] == 2

    # v2: 25 rings
    sliced_2, struct_2 = slice_rings_by_version(mat_v0, 2)
    assert sliced_2.shape == (20, 25)
    assert sum(struct_2.values()) == 25
    assert struct_2["PS"] == 2
    assert struct_2["EM1"] == 16
    assert struct_2["HAD1"] == 1
    assert struct_2["HAD2"] == 1
    assert struct_2["HAD3"] == 1

    # Invalid version raises ValueError
    with pytest.raises(ValueError):
        slice_rings_by_version(mat_v0, 99)


def test_compute_sp_with_threshold():
    """Verify SP calculation returns valid threshold."""
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.concatenate([np.linspace(0.01, 0.4, 50), np.linspace(0.6, 0.99, 50)])
    sp, pd_k, fa_k, thresh, idx = compute_sp_with_threshold(y_true, y_pred)
    assert sp > 0.95
    assert 0.35 <= thresh <= 0.65
    assert isinstance(idx, int)


def test_artifact_exporter_hierarchy(tmp_path):
    """Verify Jubiaba directory structure and file saving."""
    exporter = ArtifactExporter(
        export_dir=str(tmp_path),
        dataset_prefix="test_prefix",
        model_name="TestCNN",
        iet=5,
        ieta=0,
        version=2,
    )

    expected_base = os.path.join(
        str(tmp_path), "test_prefix", "TestCNN", "iet5", "ieta0", "v2"
    )
    assert exporter.base_dir == expected_base
    for sub in ["saved_models", "plots", "history", "metadata", "evaluation"]:
        assert os.path.isdir(os.path.join(expected_base, sub))

    # Build small model for v2 (25 rings)
    model = build_compact_cnn1d_fp32(input_dim=25, num_filters=4, kernel_size=3, dense_units=4)
    fold_path = exporter.save_fold_model(model, fold=1, repeat=1)
    assert os.path.isfile(fold_path)
    assert fold_path.endswith("model_v2_f1_r1.keras")

    best_paths = exporter.save_best_model(model, input_dim=25)
    assert os.path.isfile(best_paths["keras_path"])
    assert os.path.isfile(best_paths["onnx_path"])

    hist = {
        "loss": [0.5, 0.3],
        "val_loss": [0.45, 0.28],
        "accuracy": [0.8, 0.9],
        "val_accuracy": [0.82, 0.91],
        "val_sp": [0.81, 0.90],
    }
    hist_paths = exporter.save_history(hist)
    assert os.path.isfile(hist_paths["json_path"])
    assert os.path.isfile(hist_paths["csv_path"])

    cv_df = pd.DataFrame([
        {"fold": 1, "sp_max": 0.95, "pd_knee": 0.96, "fa_knee": 0.04, "auc": 0.99},
        {"fold": 2, "sp_max": 0.94, "pd_knee": 0.95, "fa_knee": 0.05, "auc": 0.98},
    ])
    meta_paths = exporter.save_metadata(
        training_results=[{"fold": 1, "max_sp": 0.95, "keras_history": hist}],
        cv_df=cv_df,
        kkt_result={"bit_allocation": {"conv1d_1": 4, "dense_1": 6}},
        extra_config={"rings": 25},
    )
    for p in meta_paths.values():
        assert os.path.isfile(p)

    with open(meta_paths["repro_path"], "r", encoding="utf-8") as f:
        repro = json.load(f)
    assert repro["version"] == 2
    assert repro["iet"] == 5

    x_test = np.zeros((10, 25), dtype=np.float32)
    y_test = np.array([0, 1] * 5, dtype=np.int32)
    y_pred = np.linspace(0.1, 0.9, 10, dtype=np.float32)
    eval_paths = exporter.save_evaluation_splits(x_test, y_test, y_pred, threshold=0.48)
    assert os.path.isfile(eval_paths["test_set_path"])
    assert os.path.isfile(eval_paths["predictions_path"])


def test_all_visualizations(tmp_path):
    """Verify all 8 visualization functions generate non-empty PNG files."""
    rng = np.random.default_rng(123)
    sig = np.abs(rng.normal(0.04, 0.02, size=(40, 25)))
    bkg = np.abs(rng.normal(0.04, 0.03, size=(40, 25)))
    _, struct_v2 = slice_rings_by_version(np.zeros((1, 100)), 2)

    p1 = plot_rings_profile(
        sig, bkg, struct_v2, version=2, iet=5, ieta=0,
        save_path=str(tmp_path / "rings_profile.png")
    )
    assert os.path.isfile(p1) and os.path.getsize(p1) > 0

    hist = {
        "loss": [0.5, 0.3, 0.2],
        "val_loss": [0.48, 0.32, 0.22],
        "accuracy": [0.75, 0.88, 0.92],
        "val_accuracy": [0.76, 0.87, 0.91],
        "val_sp": [0.74, 0.86, 0.91],
        "val_pd_knee": [0.80, 0.90, 0.94],
        "val_fa_knee": [0.25, 0.12, 0.08],
    }
    p2 = plot_learning_curves(hist, save_path=str(tmp_path / "learning_curves.png"))
    assert os.path.isfile(p2) and os.path.getsize(p2) > 0

    y_true = np.array([0] * 100 + [1] * 100)
    y_pred = np.concatenate([rng.uniform(0.0, 0.3, 100), rng.uniform(0.7, 1.0, 100)])

    p3 = plot_best_model_roc(
        y_true, y_pred, sp_val=0.96, pd_knee=0.97, fa_knee=0.04,
        save_path=str(tmp_path / "roc_best.png")
    )
    assert os.path.isfile(p3) and os.path.getsize(p3) > 0

    p4 = plot_cv_roc_envelope(
        [y_true, y_true], [y_pred, np.clip(y_pred + 0.02, 0, 1)],
        save_path=str(tmp_path / "roc_cv.png")
    )
    assert os.path.isfile(p4) and os.path.getsize(p4) > 0

    p5 = plot_output_score_histogram(
        y_true, y_pred, threshold_knee=0.5,
        save_path=str(tmp_path / "scores.png")
    )
    assert os.path.isfile(p5) and os.path.getsize(p5) > 0

    p6 = plot_confusion_matrix_knee(
        y_true, y_pred, threshold_knee=0.5,
        save_path=str(tmp_path / "cm.png")
    )
    assert os.path.isfile(p6) and os.path.getsize(p6) > 0

    cv_df = pd.DataFrame([
        {"sp_max": 0.95, "pd_knee": 0.96, "fa_knee": 0.04},
        {"sp_max": 0.96, "pd_knee": 0.97, "fa_knee": 0.03},
        {"sp_max": 0.94, "pd_knee": 0.95, "fa_knee": 0.05},
    ])
    p7 = plot_cv_statistics_boxplots(cv_df, save_path=str(tmp_path / "cv_box.png"))
    assert os.path.isfile(p7) and os.path.getsize(p7) > 0

    p8 = plot_fa_per_fold_workpoints(
        [y_true, y_true], [y_pred, y_pred],
        save_path=str(tmp_path / "fa_wp.png")
    )
    assert os.path.isfile(p8) and os.path.getsize(p8) > 0

    p9 = plot_roc_duel_fp32_vs_kkt(
        y_true, y_pred, 0.96, 0.97, 0.04,
        y_pred, 0.95, 0.96, 0.05,
        0.5,
        save_path=str(tmp_path / "roc_duel.png")
    )
    assert os.path.isfile(p9) and os.path.getsize(p9) > 0

    hw_8b = {
        "vu13p_resources": {"DSP": 0.0, "LUT": 24500.0, "FF": 8800.0, "BRAM_18K": 1.0},
        "breakdown": [
            {"layer_name": "conv1d_1", "LUT": 9500.0, "b_w": 8, "b_a_in": 8},
            {"layer_name": "dense_1", "LUT": 14800.0, "b_w": 8, "b_a_in": 8},
            {"layer_name": "output_dense", "LUT": 200.0, "b_w": 8, "b_a_in": 8},
        ],
    }
    hw_kkt = {
        "vu13p_resources": {"DSP": 8.0, "LUT": 10000.0, "FF": 3650.0, "BRAM_18K": 1.0},
        "breakdown": [
            {"layer_name": "conv1d_1", "LUT": 3200.0, "b_w": 4, "b_a_in": 8},
            {"layer_name": "dense_1", "LUT": 6600.0, "b_w": 6, "b_a_in": 8},
            {"layer_name": "output_dense", "LUT": 200.0, "b_w": 16, "b_a_in": 16},
        ],
    }
    budget = {"DSP": 73.7, "LUT": 10368.0, "FF": 20736.0}
    p10 = plot_vu13p_resource_utilization(hw_8b, hw_kkt, budget, save_path=str(tmp_path / "vu13p_res.png"))
    assert os.path.isfile(p10) and os.path.getsize(p10) > 0
