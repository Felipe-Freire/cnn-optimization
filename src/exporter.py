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
from sklearn.metrics import roc_curve
import tensorflow as tf
import tf_keras as keras

from src.hardware import VU13P_LIMITS, compute_model_hardware_costs
from src.models import build_compact_qcnn1d


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

    def save_region_numerical_csvs(
        self,
        cv_df: pd.DataFrame,
        kkt_result: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        folds_y_true: Optional[List[np.ndarray]] = None,
        folds_y_pred: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, str]:
        """Saves comprehensive numerical CSV tables for the trained/evaluated region.

        Generated CSV files:
        1. `metadata/vu13p_qtools_layer_breakdown.csv`
        2. `metadata/vu13p_hardware_summary.csv`
        3. `metadata/kkt_primal_dual_history.csv`
        4. `evaluation/region_evaluation_summary.csv`
        5. `evaluation/fa_workpoints_per_fold.csv`
        """
        extra_config = extra_config or {}
        kkt_result = kkt_result or {}
        csv_paths: Dict[str, str] = {}

        hw_kkt = extra_config.get("hardware_qkeras_kkt", {})
        hw_fp32 = extra_config.get("hardware_fp32", {})
        chip_limits = extra_config.get("vu13p_chip_limits", VU13P_LIMITS)
        inst_budget = (
            extra_config.get("vu13p_instance_budget")
            or kkt_result.get("vu13p_instance_budget")
            or chip_limits
        )
        cont_bits = kkt_result.get("continuous_bits", {})

        # If hardware_qkeras_kkt breakdown lacks QTools fields, compute live from saved bit allocation
        breakdown = hw_kkt.get("breakdown", [])
        if breakdown and "ap_fixed_acc" not in breakdown[0]:
            w_bits = kkt_result.get("integer_bits") or {
                item["layer_name"]: item["b_w"] for item in breakdown
            }
            a_bits = kkt_result.get("activation_bits") or {
                item["layer_name"]: item["b_a"] for item in breakdown if item["layer_name"] != "output_dense"
            }
            rings_struct = extra_config.get("rings_structure", {})
            in_dim = sum(rings_struct.values()) if rings_struct else (25 if self.version == 2 else (50 if self.version == 1 else 100))
            qmodel_tmp = build_compact_qcnn1d(
                input_dim=in_dim,
                weights_bits=w_bits,
                act_bits=a_bits,
            )
            hw_kkt = compute_model_hardware_costs(
                qmodel_tmp, weights_bits=w_bits, act_bits=a_bits, input_dim=in_dim
            )
            breakdown = hw_kkt.get("breakdown", [])

        if hw_fp32 and "total_energy_pj" not in hw_fp32:
            rings_struct = extra_config.get("rings_structure", {})
            in_dim = sum(rings_struct.values()) if rings_struct else (25 if self.version == 2 else (50 if self.version == 1 else 100))
            w_8b = {"conv1d_1": 8, "dense_1": 8, "output_dense": 8}
            a_8b = {"conv1d_1": 8, "dense_1": 8}
            qmodel_8b = build_compact_qcnn1d(input_dim=in_dim, weights_bits=w_8b, act_bits=a_8b)
            hw_fp32 = compute_model_hardware_costs(qmodel_8b, weights_bits=w_8b, act_bits=a_8b, input_dim=in_dim)

        # 1. Layer Breakdown CSV (metadata/vu13p_qtools_layer_breakdown.csv)
        if breakdown:
            layer_rows = []
            for item in breakdown:
                lname = item["layer_name"]
                layer_rows.append({
                    "iet": self.iet,
                    "ieta": self.ieta,
                    "version": f"v{self.version}",
                    "layer_name": lname,
                    "layer_type": item.get("layer_type", ""),
                    "units": item.get("units", 0),
                    "params": item.get("params", 0),
                    "macs": item.get("macs", 0),
                    "kkt_continuous_bits": round(float(cont_bits.get(lname, item.get("b_w", 8))), 4),
                    "w_bits": item.get("b_w", 8),
                    "a_bits": item.get("b_a", 8),
                    "mult_bits": item.get("mult_bits", item.get("b_w", 8) + item.get("b_a", 8)),
                    "acc_bits": item.get("acc_bits", 20),
                    "acc_int_bits": item.get("acc_int_bits", 8),
                    "ap_fixed_weight": item.get("ap_fixed_weight", ""),
                    "ap_fixed_acc": item.get("ap_fixed_acc", ""),
                    "bops": item.get("bops", 0),
                    "vu13p_DSP": float(item.get("DSP", 0.0)),
                    "vu13p_LUT": round(float(item.get("LUT", 0.0)), 2),
                    "vu13p_LUT_adder_tree_rf1": round(float(item.get("LUT_adder_tree_rf", 0.0)), 2),
                    "vu13p_FF": round(float(item.get("FF", 0.0)), 2),
                    "vu13p_FF_line_buffer": round(float(item.get("FF_line_buffer", 0.0)), 2),
                    "vu13p_BRAM_bits": float(item.get("BRAM_bits", 0.0)),
                    "energy_pj": round(float(item.get("energy_pj", 0.0)), 2),
                    "latency_cycles": int(item.get("latency_cycles", 0)),
                    "latency_ns": round(float(item.get("latency_ns", 0.0)), 2),
                })
            layer_csv_path = os.path.join(self.metadata_dir, "vu13p_qtools_layer_breakdown.csv")
            pd.DataFrame(layer_rows).to_csv(layer_csv_path, index=False)
            csv_paths["layer_breakdown_csv"] = layer_csv_path

        # 2. Hardware Summary CSV (metadata/vu13p_hardware_summary.csv)
        if hw_kkt:
            res_kkt = hw_kkt.get("vu13p_resources", {})
            res_fp32 = hw_fp32.get("vu13p_resources", {})
            final_mu = kkt_result.get("final_multipliers_mu", {})
            summary_rows = []
            for r_key in ["DSP", "LUT", "FF", "BRAM_bits"]:
                c_lim = float(chip_limits.get(r_key, 0.0))
                b_lim = float(inst_budget.get(r_key, c_lim))
                u_kkt = float(res_kkt.get(r_key, 0.0))
                u_fp32 = float(res_fp32.get(r_key, 0.0))
                slack_g = u_kkt - b_lim
                util_inst_pct = (u_kkt / b_lim * 100.0) if b_lim > 0 else (0.0 if u_kkt == 0 else 100.0)
                util_chip_pct = (u_kkt / c_lim * 100.0) if c_lim > 0 else 0.0
                summary_rows.append({
                    "iet": self.iet,
                    "ieta": self.ieta,
                    "version": f"v{self.version}",
                    "metric": r_key,
                    "vu13p_chip_limit": c_lim,
                    "instance_budget_Bj": b_lim,
                    "fp32_usage": round(u_fp32, 2),
                    "kkt_optimal_usage": round(u_kkt, 2),
                    "slack_gj": round(slack_g, 2),
                    "instance_utilization_pct": round(util_inst_pct, 2),
                    "chip_utilization_pct": round(util_chip_pct, 6),
                    "shadow_price_mu_final": round(float(final_mu.get(r_key, 0.0)), 6),
                    "feasible": bool(slack_g <= 1e-9),
                })

            for extra_metric, kkt_val, fp32_val in [
                ("total_bops", hw_kkt.get("total_bops", 0), hw_fp32.get("total_bops", 0)),
                ("total_macs", hw_kkt.get("total_macs", 0), hw_fp32.get("total_macs", 0)),
                ("total_params", hw_kkt.get("total_params", 0), hw_fp32.get("total_params", 0)),
                ("total_weight_bits", hw_kkt.get("total_weight_bits", 0), hw_fp32.get("total_weight_bits", 0)),
                ("total_energy_core_pj", hw_kkt.get("total_energy_pj", 0.0), hw_fp32.get("total_energy_pj", 0.0)),
                ("total_energy_with_io_pj", hw_kkt.get("total_energy_with_io_pj", 0.0), hw_fp32.get("total_energy_with_io_pj", 0.0)),
                ("total_latency_cycles", hw_kkt.get("total_latency_cycles", 0), hw_fp32.get("total_latency_cycles", 0)),
                ("total_latency_ns", hw_kkt.get("total_latency_ns", 0.0), hw_fp32.get("total_latency_ns", 0.0)),
            ]:
                summary_rows.append({
                    "iet": self.iet,
                    "ieta": self.ieta,
                    "version": f"v{self.version}",
                    "metric": extra_metric,
                    "vu13p_chip_limit": np.nan,
                    "instance_budget_Bj": np.nan,
                    "fp32_usage": round(float(fp32_val or 0.0), 2),
                    "kkt_optimal_usage": round(float(kkt_val or 0.0), 2),
                    "slack_gj": np.nan,
                    "instance_utilization_pct": np.nan,
                    "chip_utilization_pct": np.nan,
                    "shadow_price_mu_final": np.nan,
                    "feasible": True,
                })

            hw_sum_csv_path = os.path.join(self.metadata_dir, "vu13p_hardware_summary.csv")
            pd.DataFrame(summary_rows).to_csv(hw_sum_csv_path, index=False)
            csv_paths["hardware_summary_csv"] = hw_sum_csv_path

        # 3. Primal-Dual KKT History CSV (metadata/kkt_primal_dual_history.csv)
        pd_history = kkt_result.get("primal_dual_history", [])
        if pd_history:
            pd_rows = []
            for step in pd_history:
                w_map = step.get("w_bits", {})
                a_map = step.get("a_bits", {})
                res_map = step.get("resources", {})
                g_raw_map = step.get("g_raw", {})
                g_val_map = step.get("g_val", {})
                mu_map = step.get("mu", {})
                diag_map = step.get("kkt_diagnostics", {})
                pd_rows.append({
                    "iet": self.iet,
                    "ieta": self.ieta,
                    "version": f"v{self.version}",
                    "iteration": step.get("iteration"),
                    "w_conv1d_1": w_map.get("conv1d_1"),
                    "w_dense_1": w_map.get("dense_1"),
                    "w_output_dense": w_map.get("output_dense"),
                    "a_conv1d_1": a_map.get("conv1d_1"),
                    "a_dense_1": a_map.get("dense_1"),
                    "sp_val": round(float(step.get("sp_val", 0.0)), 6),
                    "sp_loss": round(float(step.get("sp_loss", 0.0)), 6),
                    "lagrangian_cost": round(float(step.get("lagrangian_cost", 0.0)), 6),
                    "DSP": float(res_map.get("DSP", 0.0)),
                    "LUT": round(float(res_map.get("LUT", 0.0)), 2),
                    "FF": round(float(res_map.get("FF", 0.0)), 2),
                    "BRAM_bits": float(res_map.get("BRAM_bits", 0.0)),
                    "g_raw_DSP": round(float(g_raw_map.get("DSP", 0.0)), 2),
                    "g_raw_LUT": round(float(g_raw_map.get("LUT", 0.0)), 2),
                    "g_raw_FF": round(float(g_raw_map.get("FF", 0.0)), 2),
                    "g_raw_BRAM_bits": round(float(g_raw_map.get("BRAM_bits", 0.0)), 2),
                    "g_norm_DSP": round(float(g_val_map.get("DSP", 0.0)), 6),
                    "g_norm_LUT": round(float(g_val_map.get("LUT", 0.0)), 6),
                    "g_norm_FF": round(float(g_val_map.get("FF", 0.0)), 6),
                    "g_norm_BRAM_bits": round(float(g_val_map.get("BRAM_bits", 0.0)), 6),
                    "mu_DSP": round(float(mu_map.get("DSP", 0.0)), 6),
                    "mu_LUT": round(float(mu_map.get("LUT", 0.0)), 6),
                    "mu_FF": round(float(mu_map.get("FF", 0.0)), 6),
                    "mu_BRAM_bits": round(float(mu_map.get("BRAM_bits", 0.0)), 6),
                    "primal_feasible": bool(diag_map.get("primal_feasible", False)),
                    "complementary_slackness": bool(diag_map.get("complementary_slackness", False)),
                    "kkt_converged": bool(diag_map.get("converged", False)),
                })
            kkt_hist_csv_path = os.path.join(self.metadata_dir, "kkt_primal_dual_history.csv")
            pd.DataFrame(pd_rows).to_csv(kkt_hist_csv_path, index=False)
            csv_paths["kkt_history_csv"] = kkt_hist_csv_path

        # 4. Workpoints FA per fold CSV (evaluation/fa_workpoints_per_fold.csv)
        wp_targets = [0.99, 0.985, 0.98, 0.975, 0.96, 0.95]
        wp_rows = []
        if folds_y_true and folds_y_pred and len(folds_y_true) == len(folds_y_pred):
            for f_idx, (yt, yp) in enumerate(zip(folds_y_true, folds_y_pred), start=1):
                fa_c, pd_c, _ = roc_curve(yt, yp)
                row_wp: Dict[str, Any] = {
                    "iet": self.iet,
                    "ieta": self.ieta,
                    "version": f"v{self.version}",
                    "fold": str(f_idx),
                }
                for target_pd in wp_targets:
                    valid_idx = np.where(pd_c >= target_pd)[0]
                    fa_at_wp = float(fa_c[valid_idx[0]] * 100.0) if len(valid_idx) > 0 else 100.0
                    col_lbl = str(round(target_pd * 100, 1)).replace(".0", "").replace(".", "_")
                    row_wp[f"fa_pct_at_pd_{col_lbl}"] = round(fa_at_wp, 4)
                wp_rows.append(row_wp)
        elif os.path.isfile(os.path.join(self.evaluation_dir, "holdout_test_set.npz")):
            # Compute from saved fold models and holdout test set if fold predictions weren't passed in memory
            test_npz = np.load(os.path.join(self.evaluation_dir, "holdout_test_set.npz"))
            x_eval, y_eval = test_npz["X_test"], test_npz["y_test"]
            w_bits = kkt_result.get("integer_bits")
            a_bits = kkt_result.get("activation_bits", 8)
            in_dim = x_eval.shape[-1]

            if cv_df is not None and not cv_df.empty and "model_path" in cv_df and w_bits:
                qmodel_eval = build_compact_qcnn1d(
                    input_dim=in_dim, weights_bits=w_bits, act_bits=a_bits
                )
                for _, cv_row in cv_df.iterrows():
                    m_file = os.path.join(self.saved_models_dir, str(cv_row["model_path"]))
                    if os.path.isfile(m_file):
                        qmodel_eval.load_weights(m_file)
                        yp_fold = qmodel_eval.predict(x_eval, batch_size=2048, verbose=0).ravel()
                        fa_c, pd_c, _ = roc_curve(y_eval, yp_fold)
                        row_wp = {
                            "iet": self.iet,
                            "ieta": self.ieta,
                            "version": f"v{self.version}",
                            "fold": str(int(cv_row["fold"])),
                        }
                        for target_pd in wp_targets:
                            valid_idx = np.where(pd_c >= target_pd)[0]
                            fa_at_wp = float(fa_c[valid_idx[0]] * 100.0) if len(valid_idx) > 0 else 100.0
                            col_lbl = str(round(target_pd * 100, 1)).replace(".0", "").replace(".", "_")
                            row_wp[f"fa_pct_at_pd_{col_lbl}"] = round(fa_at_wp, 4)
                        wp_rows.append(row_wp)

            preds_npz_path = os.path.join(self.evaluation_dir, "holdout_predictions.npz")
            if os.path.isfile(preds_npz_path):
                preds_npz = np.load(preds_npz_path)
                yt_ho, yp_ho = preds_npz["y_true"], preds_npz["y_pred"]
                fa_c, pd_c, _ = roc_curve(yt_ho, yp_ho)
                row_wp = {
                    "iet": self.iet,
                    "ieta": self.ieta,
                    "version": f"v{self.version}",
                    "fold": "holdout_best_model",
                }
                for target_pd in wp_targets:
                    valid_idx = np.where(pd_c >= target_pd)[0]
                    fa_at_wp = float(fa_c[valid_idx[0]] * 100.0) if len(valid_idx) > 0 else 100.0
                    col_lbl = str(round(target_pd * 100, 1)).replace(".0", "").replace(".", "_")
                    row_wp[f"fa_pct_at_pd_{col_lbl}"] = round(fa_at_wp, 4)
                wp_rows.append(row_wp)

        if wp_rows:
            wp_csv_path = os.path.join(self.evaluation_dir, "fa_workpoints_per_fold.csv")
            pd.DataFrame(wp_rows).to_csv(wp_csv_path, index=False)
            csv_paths["fa_workpoints_csv"] = wp_csv_path

        # 5. Region Consolidated Evaluation Summary CSV (evaluation/region_evaluation_summary.csv)
        ho_metrics = extra_config.get("holdout_metrics", {})
        res_kkt = hw_kkt.get("vu13p_resources", {}) if hw_kkt else {}
        region_row = {
            "dataset_prefix": self.dataset_prefix,
            "model_name": self.model_name,
            "iet": self.iet,
            "ieta": self.ieta,
            "version": f"v{self.version}",
            "n_folds": len(cv_df) if cv_df is not None else 0,
            "cv_sp_mean_pct": round(float(cv_df["sp_max"].mean() * 100.0), 4) if (cv_df is not None and "sp_max" in cv_df) else np.nan,
            "cv_sp_std_pct": round(float(cv_df["sp_max"].std() * 100.0), 4) if (cv_df is not None and "sp_max" in cv_df and len(cv_df) > 1) else 0.0,
            "cv_pd_mean_pct": round(float(cv_df["pd_knee"].mean() * 100.0), 4) if (cv_df is not None and "pd_knee" in cv_df) else np.nan,
            "cv_fa_mean_pct": round(float(cv_df["fa_knee"].mean() * 100.0), 4) if (cv_df is not None and "fa_knee" in cv_df) else np.nan,
            "cv_auc_mean": round(float(cv_df["auc"].mean()), 6) if (cv_df is not None and "auc" in cv_df) else np.nan,
            "holdout_kkt_sp_pct": round(float(ho_metrics.get("sp", 0.0)) * 100.0, 4) if ho_metrics else np.nan,
            "holdout_kkt_pd_pct": round(float(ho_metrics.get("pd", 0.0)) * 100.0, 4) if ho_metrics else np.nan,
            "holdout_kkt_fa_pct": round(float(ho_metrics.get("fa", 0.0)) * 100.0, 4) if ho_metrics else np.nan,
            "holdout_kkt_threshold": round(float(ho_metrics.get("threshold", 0.5)), 6) if ho_metrics else np.nan,
            "holdout_fp32_sp_pct": round(float(ho_metrics.get("fp32_sp", 0.0)) * 100.0, 4) if ho_metrics else np.nan,
            "holdout_fp32_pd_pct": round(float(ho_metrics.get("fp32_pd", 0.0)) * 100.0, 4) if ho_metrics else np.nan,
            "holdout_fp32_fa_pct": round(float(ho_metrics.get("fp32_fa", 0.0)) * 100.0, 4) if ho_metrics else np.nan,
            "vu13p_DSP": float(res_kkt.get("DSP", 0.0)),
            "vu13p_LUT": round(float(res_kkt.get("LUT", 0.0)), 2),
            "vu13p_FF": round(float(res_kkt.get("FF", 0.0)), 2),
            "vu13p_BRAM_bits": float(res_kkt.get("BRAM_bits", 0.0)),
            "total_bops": int(hw_kkt.get("total_bops", 0)) if hw_kkt else 0,
            "total_energy_core_pj": round(float(hw_kkt.get("total_energy_pj", 0.0)), 2) if hw_kkt else 0.0,
            "total_latency_ns": round(float(hw_kkt.get("total_latency_ns", 0.0)), 2) if hw_kkt else 0.0,
        }
        reg_eval_csv_path = os.path.join(self.evaluation_dir, "region_evaluation_summary.csv")
        pd.DataFrame([region_row]).to_csv(reg_eval_csv_path, index=False)
        csv_paths["region_evaluation_summary_csv"] = reg_eval_csv_path

        return csv_paths

    def save_metadata(
        self,
        training_results: List[Dict[str, Any]],
        cv_df: pd.DataFrame,
        kkt_result: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        folds_y_true: Optional[List[np.ndarray]] = None,
        folds_y_pred: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, str]:
        """Saves training metadata (.pkl/.json), CV results (.csv), reproducibility info, and numerical CSV tables."""
        pkl_path = os.path.join(
            self.metadata_dir, f"training_metadata_v{self.version}.pkl"
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(training_results, f)

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

        csv_table_paths = self.save_region_numerical_csvs(
            cv_df=cv_df,
            kkt_result=kkt_result,
            extra_config=extra_config,
            folds_y_true=folds_y_true,
            folds_y_pred=folds_y_pred,
        )
        paths.update(csv_table_paths)

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


def export_csvs_for_existing_regions(
    results_root: str = "results",
) -> Dict[str, Dict[str, str]]:
    """Scans all trained/evaluated region directories under `results_root` and generates CSV reports.

    Does not re-train any models. Reads existing `reproducibility_metadata.json`,
    `kkt_allocation_results.json`, `cross_validation_results.csv`, and `holdout_predictions.npz`
    for every region (`iet{iet}/ieta{ieta}/v{version}`) and writes the standardized CSV tables.

    Args:
        results_root: Root directory containing Jubiaba experiment outputs.

    Returns:
        Mapping from region relative path to the dictionary of generated CSV file paths.
    """
    exported_regions: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(results_root):
        return exported_regions

    for dirpath, _, filenames in os.walk(results_root):
        if "reproducibility_metadata.json" in filenames and os.path.basename(dirpath) == "metadata":
            region_base = os.path.dirname(dirpath)
            repro_path = os.path.join(dirpath, "reproducibility_metadata.json")
            kkt_path = os.path.join(dirpath, "kkt_allocation_results.json")
            cv_csv_path = os.path.join(dirpath, "cross_validation_results.csv")

            with open(repro_path, "r", encoding="utf-8") as f:
                repro = json.load(f)

            kkt_res = None
            if os.path.isfile(kkt_path):
                with open(kkt_path, "r", encoding="utf-8") as f:
                    kkt_res = json.load(f)

            cv_df = pd.read_csv(cv_csv_path) if os.path.isfile(cv_csv_path) else pd.DataFrame()

            exporter = ArtifactExporter(
                export_dir=results_root,
                dataset_prefix=repro.get("dataset_prefix", "mc21_13p6TeV.Zee.JF17.2sigma.5M"),
                model_name=repro.get("model_name", "QCNN1D_KKT"),
                iet=int(repro.get("iet", 5)),
                ieta=int(repro.get("ieta", 0)),
                version=int(repro.get("version", 2)),
            )

            csv_paths = exporter.save_region_numerical_csvs(
                cv_df=cv_df,
                kkt_result=kkt_res,
                extra_config=repro.get("extra_config", {}),
            )
            rel_region = os.path.relpath(region_base, results_root)
            exported_regions[rel_region] = csv_paths

    return exported_regions
