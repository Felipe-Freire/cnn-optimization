"""Metrics, ring slicing, and callbacks for ATLAS L0 Ringer optimization."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_curve
import tensorflow as tf
import tf_keras as keras
from tf_keras.callbacks import Callback

ESTRUTURA_BASE: Dict[str, int] = {
    "PS": 8,
    "EM1": 64,
    "EM2": 8,
    "EM3": 8,
    "HAD1": 4,
    "HAD2": 4,
    "HAD3": 4,
}

CONFIG_VERSOES: Dict[int, Dict[str, int]] = {
    0: {"input_dim": 100, "fator": 1, "kernel": 5, "filters": 8, "dense_units": 8},
    1: {"input_dim": 50, "fator": 2, "kernel": 3, "filters": 8, "dense_units": 8},
    2: {"input_dim": 25, "fator": 4, "kernel": 3, "filters": 8, "dense_units": 8},
}


def slice_rings_by_version(
    matriz_v0: np.ndarray, version: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Slices a 100-ring calorimeter matrix according to version (v0=100, v1=50, v2=25).

    Retains the initial inner rings of each calorimeter sampling layer, which hold the
    highest signal-to-noise ratio around the shower barycenter.

    Args:
        matriz_v0: Array of shape (N, 100) containing raw or normalized ring features.
        version: Ring version (0 -> 100 rings, 1 -> 50 rings, 2 -> 25 rings).

    Returns:
        Tuple of (sliced_matrix, structure_dict).
    """
    if version not in CONFIG_VERSOES:
        raise ValueError(f"Unsupported version {version}. Choose 0 (100), 1 (50), or 2 (25).")

    fator = CONFIG_VERSOES[version]["fator"]
    if fator == 1:
        return matriz_v0.copy(), ESTRUTURA_BASE.copy()

    ponteiro = 0
    indices_manter = []
    estrutura_reduzida = {}
    for camada, qtd_original in ESTRUTURA_BASE.items():
        qtd_reduzida = qtd_original // fator
        indices_manter.extend(range(ponteiro, ponteiro + qtd_reduzida))
        estrutura_reduzida[camada] = qtd_reduzida
        ponteiro += qtd_original

    return matriz_v0[:, indices_manter], estrutura_reduzida


def norm1(data: np.ndarray) -> np.ndarray:
    """Normalize calorimeter ring energy per event (L1 norm).

    Args:
        data: Array of shape (N, M) containing raw ring energies.

    Returns:
        Normalized array where each row sums to 1 (or remains 0 if all 0).
    """
    norms = np.abs(data.sum(axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return data / norms


def compute_sp(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, int]:
    """Calculate the Sum-Product (SP) index at the ROC knee point.

    SP = sqrt( sqrt(PD * (1 - FA)) * 0.5 * (PD + (1 - FA)) )

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted probabilities/scores.

    Returns:
        Tuple of (max_sp, pd_at_knee, fa_at_knee, knee_index).
    """
    fa, pd, _ = roc_curve(y_true, y_pred)
    if fa.size == 0 or pd.size == 0 or np.all(np.isnan(fa)) or np.all(np.isnan(pd)):
        return 0.0, 0.0, 1.0, 0

    term_prod = np.sqrt(np.clip(pd * (1.0 - fa), 0.0, 1.0))
    term_sum = 0.5 * (pd + (1.0 - fa))
    sp_curve = np.sqrt(term_prod * term_sum)

    knee = int(np.nanargmax(sp_curve))
    max_sp = float(sp_curve[knee])
    pd_knee = float(pd[knee])
    fa_knee = float(fa[knee])

    return max_sp, pd_knee, fa_knee, knee


def compute_sp_with_threshold(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, float, int]:
    """Calculate SP index, PD, FA, and exact decision threshold at the ROC knee."""
    fa, pd, thresholds = roc_curve(y_true, y_pred)
    if fa.size == 0 or pd.size == 0 or np.all(np.isnan(fa)) or np.all(np.isnan(pd)):
        return 0.0, 0.0, 1.0, 0.5, 0

    term_prod = np.sqrt(np.clip(pd * (1.0 - fa), 0.0, 1.0))
    term_sum = 0.5 * (pd + (1.0 - fa))
    sp_curve = np.sqrt(term_prod * term_sum)

    knee = int(np.nanargmax(sp_curve))
    return (
        float(sp_curve[knee]),
        float(pd[knee]),
        float(fa[knee]),
        float(thresholds[knee]),
        knee,
    )


class SPCallback(Callback):
    """Keras Callback for monitoring SP index and saving best model weights."""

    def __init__(
        self,
        val_data: Tuple[np.ndarray, np.ndarray],
        patience: int = 15,
        verbose: bool = False,
    ):
        super().__init__()
        self.val_x, self.val_y = val_data
        self.patience = patience
        self.verbose = verbose
        self.best_sp = -1.0
        self.best_epoch = 0
        self.best_weights = None
        self.best_pd_at_knee = 0.0
        self.best_fa_at_knee = 1.0
        self.best_threshold = 0.5
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if logs is None:
            logs = {}

        y_pred = self.model.predict(self.val_x, batch_size=2048, verbose=0).ravel()
        sp_val, pd_knee, fa_knee, thresh, _ = compute_sp_with_threshold(self.val_y, y_pred)

        logs["val_sp"] = sp_val
        logs["val_pd_knee"] = pd_knee
        logs["val_fa_knee"] = fa_knee

        if sp_val > self.best_sp:
            self.best_sp = sp_val
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.best_pd_at_knee = pd_knee
            self.best_fa_at_knee = fa_knee
            self.best_threshold = thresh
            self.wait = 0
            if self.verbose:
                print(
                    f"\n[SPCallback] Epoch {epoch+1}: New best SP = {sp_val:.4f} "
                    f"(PD={pd_knee:.4f}, FA={fa_knee:.4f}, Thresh={thresh:.4f})"
                )
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(
                        f"\n[SPCallback] Early stopping at epoch {epoch+1}. "
                        f"Best SP was {self.best_sp:.4f} at epoch {self.best_epoch+1}."
                    )
                self.model.stop_training = True

    def restore_best_weights(self) -> None:
        """Restores model to the best recorded SP weights."""
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
