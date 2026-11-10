"""Training helpers shared by the notebook (FP32 reference, warm-start QAT, SP evaluation)."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from typing import Dict, Optional, Tuple

import tf_keras as keras

from src.metrics import SPCallback, compute_sp_with_threshold, sp_at_threshold
from src.models import build_compact_cnn1d_fp32, build_compact_qcnn1d, transfer_weights_to_qmodel


def train_fp32(arch: Dict[str, int], X_tr, y_tr, X_val, y_val, epochs: int = 18, lr: float = 1.5e-3,
               patience: int = 8, seed: int = 42, verbose: int = 0) -> Tuple[keras.Model, SPCallback, dict]:
    keras.utils.set_random_seed(seed)
    model = build_compact_cnn1d_fp32(arch["input_dim"], arch["filters"], arch["kernel"], arch["dense_units"],
                                     learning_rate=lr)
    cb = SPCallback(val_data=(X_val, y_val), patience=patience)
    sched = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=512,
                     callbacks=[cb, sched], verbose=verbose)
    cb.restore_best_weights()
    return model, cb, hist.history


def train_qat(arch: Dict[str, int], w_bits: Dict[str, int], a_bits: Dict[str, int], input_bits: Optional[int],
              warm_start: keras.Model, X_tr, y_tr, X_val, y_val, epochs: int = 18, lr: float = 1e-3,
              patience: int = 8, seed: int = 42, verbose: int = 0) -> Tuple[keras.Model, SPCallback, dict]:
    keras.utils.set_random_seed(seed)
    model = build_compact_qcnn1d(arch["input_dim"], arch["filters"], arch["kernel"], arch["dense_units"],
                                 weights_bits=w_bits, act_bits=a_bits, input_bits=input_bits, learning_rate=lr)
    transfer_weights_to_qmodel(warm_start, model)
    cb = SPCallback(val_data=(X_val, y_val), patience=patience)
    sched = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=512,
                     callbacks=[cb, sched], verbose=verbose)
    cb.restore_best_weights()
    return model, cb, hist.history


def evaluate_at_threshold(model, X, y, threshold: float) -> Dict[str, float]:
    """SP/PD/FA at a threshold fixed beforehand (e.g. on validation) plus the knee SP for reference."""
    p = model.predict(X, batch_size=4096, verbose=0).ravel()
    sp, pd_, fa = sp_at_threshold(y, p, threshold)
    sp_knee = compute_sp_with_threshold(y, p)[0]
    return {"sp": sp, "pd": pd_, "fa": fa, "threshold": float(threshold), "sp_knee_optimistic": sp_knee,
            "y_pred": p}
