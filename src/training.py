"""Training and evaluation helpers shared by the allocation loop and the cross-validation."""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import tf_keras as keras
from sklearn.metrics import auc, roc_curve

from src.metrics import SPCallback, compute_sp_with_threshold, sp_at_threshold
from src.models import build_compact_cnn1d_fp32, build_compact_qcnn1d, transfer_weights_to_qmodel


def _fit(model, X_tr, y_tr, X_val, y_val, epochs, patience, reduce_lr):
    cb = SPCallback(val_data=(X_val, y_val), patience=patience)
    callbacks = [cb]
    if reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0
        ))
    history = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=512,
        callbacks=callbacks, verbose=0,
    )
    cb.restore_best_weights()
    return cb, history.history


def train_fp32(
    arch: Dict[str, int], X_tr, y_tr, X_val, y_val, epochs: int,
    learning_rate: float = 1.5e-3, patience: int = 8, seed: int = 42, reduce_lr: bool = True,
) -> Tuple[keras.Model, SPCallback, Dict[str, list]]:
    """FP32 reference trained on (X_tr, y_tr); the best validation-SP epoch is kept."""
    keras.utils.set_random_seed(seed)
    model = build_compact_cnn1d_fp32(
        arch["input_dim"], arch["filters"], arch["kernel"], arch["dense_units"], learning_rate=learning_rate
    )
    cb, history = _fit(model, X_tr, y_tr, X_val, y_val, epochs, patience, reduce_lr)
    return model, cb, history


def train_qat(
    arch: Dict[str, int], w_bits: Dict[str, int], a_bits: Dict[str, int], input_bits: Optional[int],
    warm_start: keras.Model, X_tr, y_tr, X_val, y_val, epochs: int,
    learning_rate: float = 1e-3, patience: int = 8, seed: int = 42, reduce_lr: bool = True,
) -> Tuple[keras.Model, SPCallback, Dict[str, list]]:
    """QAT of the QKeras model, warm-started from ``warm_start`` (an FP32 model trained on the same X_tr)."""
    keras.utils.set_random_seed(seed)
    model = build_compact_qcnn1d(
        arch["input_dim"], arch["filters"], arch["kernel"], arch["dense_units"],
        weights_bits=w_bits, act_bits=a_bits, input_bits=input_bits, learning_rate=learning_rate,
    )
    transfer_weights_to_qmodel(warm_start, model)
    cb, history = _fit(model, X_tr, y_tr, X_val, y_val, epochs, patience, reduce_lr)
    return model, cb, history


def evaluate_model(model: keras.Model, X_val, y_val, X_test, y_test) -> Dict[str, Any]:
    """Knee SP on validation, and SP/PD/FA on the test set at the threshold chosen on validation.

    ``sp_test_knee`` (knee re-optimized on the test set) is reported only as an optimistic reference.
    """
    p_val = model.predict(X_val, batch_size=4096, verbose=0).ravel()
    p_test = model.predict(X_test, batch_size=4096, verbose=0).ravel()
    sp_val, pd_val, fa_val, threshold, _ = compute_sp_with_threshold(y_val, p_val)
    sp_test, pd_test, fa_test = sp_at_threshold(y_test, p_test, threshold)
    fa_c, pd_c, _ = roc_curve(y_val, p_val)
    return {
        "sp_val": sp_val, "pd_val": pd_val, "fa_val": fa_val, "threshold": threshold,
        "auc_val": float(auc(fa_c, pd_c)),
        "sp_test": sp_test, "pd_test": pd_test, "fa_test": fa_test,
        "sp_test_knee": compute_sp_with_threshold(y_test, p_test)[0],
        "y_pred_val": p_val, "y_pred_test": p_test,
    }


def paired_difference(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Mean, standard deviation and standard error of a_k - b_k over folds trained on the same data."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    std = float(d.std(ddof=1)) if d.size > 1 else 0.0
    return {"mean": float(d.mean()), "std": std, "se": std / np.sqrt(d.size) if d.size else 0.0, "n": int(d.size)}
