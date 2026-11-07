"""Metrics and callbacks for ATLAS L0 Ringer optimization."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_curve
import tensorflow as tf
import tf_keras as keras
from tf_keras.callbacks import Callback


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
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if logs is None:
            logs = {}

        y_pred = self.model.predict(self.val_x, batch_size=2048, verbose=0).ravel()
        sp_val, pd_knee, fa_knee, _ = compute_sp(self.val_y, y_pred)

        logs["val_sp"] = sp_val
        logs["val_pd_knee"] = pd_knee
        logs["val_fa_knee"] = fa_knee

        if sp_val > self.best_sp:
            self.best_sp = sp_val
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.best_pd_at_knee = pd_knee
            self.best_fa_at_knee = fa_knee
            self.wait = 0
            if self.verbose:
                print(f"\n[SPCallback] Epoch {epoch+1}: New best SP = {sp_val:.4f} (PD={pd_knee:.4f}, FA={fa_knee:.4f})")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\n[SPCallback] Early stopping at epoch {epoch+1}. Best SP was {self.best_sp:.4f} at epoch {self.best_epoch+1}.")
                self.model.stop_training = True

    def restore_best_weights(self) -> None:
        """Restores model to the best recorded SP weights."""
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
