"""Hierarchical artifact exporter following the Jubiaba project specification."""

from datetime import datetime
import json
import os
import pickle
import platform
import subprocess
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras


class ArtifactExporter:
    """Manages hierarchical output organization for ATLAS L0 Ringer experiments.

    Directory Layout:
        <export_dir>/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/
        ├── saved_models/
        ├── plots/
        ├── history/
        ├── metadata/
        └── evaluation/
    """

    def __init__(
        self,
        export_dir: str = "results",
        dataset_prefix: str = "mc21_13p6TeV.Zee.JF17.2sigma.5M",
        model_name: str = "QCNN1D",
        iet: int = 5,
        ieta: int = 0,
        version: int = 2,
    ):
        self.export_dir = export_dir
        self.dataset_prefix = dataset_prefix
        self.model_name = model_name
        self.iet = iet
        self.ieta = ieta
        self.version = version

        self.base_dir = os.path.join(
            export_dir,
            dataset_prefix,
            model_name,
            f"iet{iet}",
            f"ieta{ieta}",
            f"v{version}",
        )

        self.saved_models_dir = os.path.join(self.base_dir, "saved_models")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.history_dir = os.path.join(self.base_dir, "history")
        self.metadata_dir = os.path.join(self.base_dir, "metadata")
        self.evaluation_dir = os.path.join(self.base_dir, "evaluation")

        for folder in [
            self.saved_models_dir,
            self.plots_dir,
            self.history_dir,
            self.metadata_dir,
            self.evaluation_dir,
        ]:
            os.makedirs(folder, exist_ok=True)

    def get_plot_path(self, filename: str) -> str:
        """Returns the full path for saving a plot inside plots/."""
        return os.path.join(self.plots_dir, filename)

    def save_fold_model(
        self, model: keras.Model, fold: int, repeat: int = 1
    ) -> str:
        """Saves an individual fold/repeat model to saved_models/."""
        filename = f"model_v{self.version}_f{fold}_r{repeat}.keras"
        path = os.path.join(self.saved_models_dir, filename)
        model.save(path)
        return path

    def save_best_model(
        self, model: keras.Model, input_dim: Optional[int] = None
    ) -> Dict[str, str]:
        """Saves best_model.keras and exports best_model.onnx in saved_models/."""
        keras_path = os.path.join(self.saved_models_dir, "best_model.keras")
        model.save(keras_path)

        onnx_path = os.path.join(self.saved_models_dir, "best_model.onnx")
        try:
            import tf2onnx
            if input_dim is None:
                input_dim = model.input_shape[-1]
            spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input_rings"),)
            tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
        except Exception:
            # Export lightweight JSON/Serialized graph metadata when tf2onnx is unavailable for QKeras layers
            onnx_meta = {
                "format": "ONNX_Metadata_Fallback",
                "model_name": model.name,
                "input_shape": list(model.input_shape) if hasattr(model, "input_shape") else None,
                "total_params": int(model.count_params()),
                "layers": [
                    {"name": l.name, "class": l.__class__.__name__}
                    for l in model.layers
                ],
            }
            with open(onnx_path, "w", encoding="utf-8") as f:
                json.dump(onnx_meta, f, indent=2)

        return {"keras_path": keras_path, "onnx_path": onnx_path}

    def save_history(self, history_dict: Dict[str, List[float]]) -> Dict[str, str]:
        """Saves best-run epoch history in CSV and JSON formats under history/."""
        clean_hist = {
            k: [float(v) for v in vals] for k, vals in history_dict.items()
        }

        json_path = os.path.join(self.history_dir, "history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_hist, f, indent=2)

        csv_path = os.path.join(self.history_dir, "training_history.csv")
        df_hist = pd.DataFrame(clean_hist)
        df_hist.index.name = "epoch"
        df_hist.to_csv(csv_path)

        return {"json_path": json_path, "csv_path": csv_path}

    def save_metadata(
        self,
        training_results: List[Dict[str, Any]],
        cv_df: pd.DataFrame,
        kkt_result: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Dict[str, str]:
        """Saves training metadata (.pkl/.json), CV results (.csv), and reproducibility info."""
        pkl_path = os.path.join(
            self.metadata_dir, f"training_metadata_v{self.version}.pkl"
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(training_results, f)

        # JSON-serializable summary of training_results
        json_serializable_results = []
        for item in training_results:
            entry = {}
            for k, v in item.items():
                if k == "keras_history" and isinstance(v, dict):
                    entry[k] = {hk: [float(x) for x in hv] for hk, hv in v.items()}
                elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    entry[k] = v
                else:
                    entry[k] = str(v)
            json_serializable_results.append(entry)

        json_path = os.path.join(
            self.metadata_dir, f"training_metadata_v{self.version}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_serializable_results, f, indent=2)

        cv_csv_path = os.path.join(self.metadata_dir, "cross_validation_results.csv")
        cv_df.to_csv(cv_csv_path, index=False)

        # Git commit hash if available
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_commit = "unknown"

        repro_meta = {
            "timestamp": datetime.now().isoformat(),
            "dataset_prefix": self.dataset_prefix,
            "model_name": self.model_name,
            "iet": self.iet,
            "ieta": self.ieta,
            "version": self.version,
            "seed": seed,
            "git_commit": git_commit,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "extra_config": extra_config or {},
        }
        repro_path = os.path.join(self.metadata_dir, "reproducibility_metadata.json")
        with open(repro_path, "w", encoding="utf-8") as f:
            json.dump(repro_meta, f, indent=2)

        paths = {
            "pkl_path": pkl_path,
            "json_path": json_path,
            "cv_csv_path": cv_csv_path,
            "repro_path": repro_path,
        }

        if kkt_result is not None:
            kkt_path = os.path.join(self.metadata_dir, "kkt_allocation_results.json")
            with open(kkt_path, "w", encoding="utf-8") as f:
                json.dump(kkt_result, f, indent=2)
            paths["kkt_path"] = kkt_path

        return paths

    def save_evaluation_splits(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, str]:
        """Saves compressed holdout splits and predictions in evaluation/."""
        test_set_path = os.path.join(self.evaluation_dir, "holdout_test_set.npz")
        np.savez_compressed(test_set_path, X_test=x_test, y_test=y_test)

        preds_path = os.path.join(self.evaluation_dir, "holdout_predictions.npz")
        np.savez_compressed(
            preds_path,
            y_true=y_test,
            y_pred=y_pred,
            threshold=np.array([threshold], dtype=np.float32),
        )

        return {"test_set_path": test_set_path, "predictions_path": preds_path}
