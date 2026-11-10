"""Scientific visualization suite for ATLAS L0 Ringer (codeTCC & Jubiaba standard)."""

import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def set_academic_style() -> None:
    """Applies clean academic formatting for publication-grade figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_rings_profile(
    sig_normed: np.ndarray,
    bkg_normed: np.ndarray,
    estrutura: Dict[str, int],
    version: int,
    iet: int,
    ieta: int,
    save_path: str,
    show: bool = False,
) -> str:
    """Plots mean energy deposition profile across calorimeter rings (codeTCC style)."""
    set_academic_style()
    m_sig = np.mean(sig_normed, axis=0)
    s_sig = np.std(sig_normed, axis=0)
    m_bkg = np.mean(bkg_normed, axis=0)
    s_bkg = np.std(bkg_normed, axis=0)

    x_axis = np.arange(len(m_sig))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(
        x_axis, m_sig, s_sig,
        marker="o", color="navy", label="Elétron (Sinal)",
        ms=3.5, elinewidth=0.9, capsize=2, alpha=0.9
    )
    ax.errorbar(
        x_axis, m_bkg, s_bkg,
        marker="s", color="darkorange", label="Jatos (Fundo)",
        ms=3.5, elinewidth=0.9, capsize=2, alpha=0.9
    )

    pos = 0
    cores = ["#1b9e77", "#d95f09", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    for i, (camada, qtd) in enumerate(estrutura.items()):
        if qtd > 0:
            ax.axvline(x=pos, color=cores[i % len(cores)], linestyle="--", alpha=0.5)
            ax.text(
                pos + 0.15, 0.52, camada,
                rotation=90, color=cores[i % len(cores)],
                fontsize=9, fontweight="bold", va="bottom"
            )
            pos += qtd

    ax.set_title(
        f"Perfil Médio de Energia Normalizada - v{version} ({len(x_axis)} Anéis) | IET {iet}, IETA {ieta}",
        fontweight="bold", pad=12
    )
    ax.set_xlabel("Índice do Anel")
    ax.set_ylabel("Energia Normalizada (L1)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_ylim(-0.05, 0.72)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_learning_curves(
    history_dict: Dict[str, List[float]],
    save_path: str,
    title_prefix: str = "",
    show: bool = False,
) -> str:
    """Plots Loss, Accuracy, and SP/PD/FA evolution across training epochs."""
    set_academic_style()
    epochs = np.arange(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # 1. Loss
    axes[0].plot(epochs, history_dict["loss"], label="Treino", color="steelblue", lw=2)
    if "val_loss" in history_dict:
        axes[0].plot(epochs, history_dict["val_loss"], label="Validação", color="darkorange", lw=2)
    axes[0].set_title(f"{title_prefix} Evolução da Perda (BCE)".strip(), fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()

    # 2. Accuracy
    if "accuracy" in history_dict:
        axes[1].plot(epochs, history_dict["accuracy"], label="Treino", color="steelblue", lw=2)
    if "val_accuracy" in history_dict:
        axes[1].plot(epochs, history_dict["val_accuracy"], label="Validação", color="darkorange", lw=2)
    axes[1].set_title(f"{title_prefix} Acurácia por Época".strip(), fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()

    # 3. SP, PD, 1-FA
    if "val_sp" in history_dict:
        axes[2].plot(epochs, history_dict["val_sp"], label="Índice SP (Val)", color="purple", marker="o", ms=4, lw=2)
    if "val_pd_knee" in history_dict:
        axes[2].plot(epochs, history_dict["val_pd_knee"], label="PD no Knee", color="forestgreen", linestyle="--", lw=1.5)
    if "val_fa_knee" in history_dict:
        axes[2].plot(
            epochs, [1.0 - x for x in history_dict["val_fa_knee"]],
            label="Rejeição (1 - FA)", color="crimson", linestyle=":", lw=1.5
        )
    axes[2].set_title(f"{title_prefix} Métricas ATLAS Ringer (SP)".strip(), fontweight="bold")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("Taxa / Índice")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_best_model_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sp_val: float,
    pd_knee: float,
    fa_knee: float,
    save_path: str,
    title: str = "Curva ROC - Melhor Modelo",
    show: bool = False,
) -> str:
    """Plots full ROC and operational zoomed ROC with working points (codeTCC style)."""
    set_academic_style()
    fa, pd, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fa, pd)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Full ROC
    axes[0].plot(fa * 100, pd * 100, color="#1f4e79", lw=2.2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot(
        [fa_knee * 100], [pd_knee * 100],
        marker="*", markersize=13, color="crimson",
        label=f"Knee SP={sp_val:.4f} (PD={pd_knee*100:.2f}%, FA={fa_knee*100:.2f}%)"
    )
    axes[0].plot([0, 100], [0, 100], color="gray", linestyle="--", lw=1)
    axes[0].set_title(f"{title} (Global)", fontweight="bold")
    axes[0].set_xlabel("Taxa de Falso Alarme - FA [%]")
    axes[0].set_ylabel("Probabilidade de Detecção - PD [%]")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right")

    # Zoomed ROC (Operational Trigger Region)
    axes[1].plot(fa * 100, pd * 100, color="#1f4e79", lw=2.5, label="Curva Operacional")
    axes[1].plot(
        [fa_knee * 100], [pd_knee * 100],
        marker="*", markersize=14, color="crimson",
        label=f"Joelho SP ({sp_val:.4f})"
    )

    # Mark standard ATLAS trigger working points
    wps = [("Tight (96%)", 0.96, "o", "#1b9e77"), ("Medium (97.5%)", 0.975, "s", "#d95f02"), ("Loose (98.5%)", 0.985, "^", "#7570b3")]
    for wp_label, target_pd, marker, col in wps:
        idx = int(np.argmin(np.abs(pd - target_pd)))
        axes[1].plot(
            [fa[idx] * 100], [pd[idx] * 100],
            marker=marker, markersize=8, color=col,
            label=f"{wp_label}: FA={fa[idx]*100:.2f}%"
        )

    axes[1].set_xlim(-0.1, 6.0)
    axes[1].set_ylim(94.0, 100.1)
    axes[1].set_title(f"{title} (Zoom Região de Trigger)", fontweight="bold")
    axes[1].set_xlabel("Taxa de Falso Alarme - FA [%]")
    axes[1].set_ylabel("Probabilidade de Detecção - PD [%]")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_cv_roc_envelope(
    folds_y_true: List[np.ndarray],
    folds_y_pred: List[np.ndarray],
    save_path: str,
    title: str = "Envoltória ROC em Validação Cruzada",
    show: bool = False,
) -> str:
    """Plots mean ROC curve and +-1 std envelope across K-folds."""
    set_academic_style()
    mean_fa = np.linspace(0, 0.10, 200)
    interp_pds = []

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (yt, yp) in enumerate(zip(folds_y_true, folds_y_pred)):
        fa, pd, _ = roc_curve(yt, yp)
        interp_pd = np.interp(mean_fa, fa, pd)
        interp_pd[0] = 0.0
        interp_pds.append(interp_pd)
        ax.plot(fa * 100, pd * 100, lw=1, alpha=0.3, label=f"Fold {i+1}" if i < 5 else None)

    mean_pd = np.mean(interp_pds, axis=0)
    std_pd = np.std(interp_pds, axis=0)

    ax.plot(mean_fa * 100, mean_pd * 100, color="navy", lw=2.5, label="Média dos Folds")
    ax.fill_between(
        mean_fa * 100,
        np.clip((mean_pd - std_pd) * 100, 0, 100),
        np.clip((mean_pd + std_pd) * 100, 0, 100),
        color="steelblue", alpha=0.25, label="Envelope ±1σ"
    )

    ax.set_xlim(-0.1, 8.0)
    ax.set_ylim(92.0, 100.1)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Taxa de Falso Alarme - FA [%]")
    ax.set_ylabel("Probabilidade de Detecção - PD [%]")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_output_score_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_knee: float,
    save_path: str,
    title: str = "Distribuição das Saídas da Rede (Sinal vs Fundo)",
    show: bool = False,
) -> str:
    """Plots histogram of network output scores in linear and log scales."""
    set_academic_style()
    sig_scores = y_pred[y_true == 1]
    bkg_scores = y_pred[y_true == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0, 1, 50)

    for idx, (ax, scale_label) in enumerate(zip(axes, ["Escala Linear", "Escala Logarítmica"])):
        ax.hist(sig_scores, bins=bins, alpha=0.65, color="#1f4e79", label="Elétrons (Sinal)", density=True)
        ax.hist(bkg_scores, bins=bins, alpha=0.65, color="#a33030", label="Jatos (Fundo)", density=True)
        ax.axvline(
            threshold_knee, color="black", linestyle="--", lw=2,
            label=f"Limiar Knee ({threshold_knee:.3f})"
        )
        if idx == 1:
            ax.set_yscale("log")
        ax.set_title(f"{title} ({scale_label})", fontweight="bold")
        ax.set_xlabel("Score de Saída (Sigmoid)")
        ax.set_ylabel("Densidade")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_confusion_matrix_knee(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_knee: float,
    save_path: str,
    title: str = "Matriz de Confusão no Ponto de Joelho (SP)",
    show: bool = False,
) -> str:
    """Plots normalized confusion matrix at the SP knee threshold."""
    set_academic_style()
    y_hat = (y_pred >= threshold_knee).astype(int)
    cm = confusion_matrix(y_true, y_hat)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, format="%.1f%%")

    classes = ["Jato (Fundo: 0)", "Elétron (Sinal: 1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Classe Predita", fontweight="bold")
    ax.set_ylabel("Classe Real", fontweight="bold")
    ax.set_title(f"{title}\n(Limiar = {threshold_knee:.4f})", fontweight="bold", pad=12)

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_norm[i, j] > 50 else "black"
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}%\n({cm[i, j]:,})",
                ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_cv_statistics_boxplots(
    cv_df: pd.DataFrame,
    save_path: str,
    title: str = "Estabilidade em Validação Cruzada (K-Folds)",
    show: bool = False,
) -> str:
    """Plots boxplots of SP, PD, and FA across cross-validation folds."""
    set_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics_info = [
        ("sp_max", "Índice SP [%]", 100.0, "#7570b3"),
        ("pd_knee", "Prob. Detecção - PD [%]", 100.0, "#1b9e77"),
        ("fa_knee", "Falso Alarme - FA [%]", 100.0, "#d95f02"),
    ]

    for ax, (col, label, scale, color) in zip(axes, metrics_info):
        vals = cv_df[col].values * scale
        bp = ax.boxplot([vals], patch_artist=True, tick_labels=[label], widths=0.45)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.scatter(np.ones_like(vals), vals, color="black", zorder=3, s=25, alpha=0.8)
        ax.set_title(f"{label}\n(Média: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_fa_per_fold_workpoints(
    folds_y_true: List[np.ndarray],
    folds_y_pred: List[np.ndarray],
    save_path: str,
    title: str = "Distribuição de Falso Alarme (FA) por Ponto de Operação",
    show: bool = False,
) -> str:
    """Plots boxplot of False Alarm rate across folds for fixed PD Working Points (codeTCC style)."""
    set_academic_style()
    workpoints = {
        "Tight (PD=96%)": 0.96,
        "Medium (PD=97.5%)": 0.975,
        "Loose (PD=98.5%)": 0.985,
        "Very Loose (PD=99%)": 0.99,
    }
    fa_by_wp = {k: [] for k in workpoints}

    for yt, yp in zip(folds_y_true, folds_y_pred):
        fa, pd, _ = roc_curve(yt, yp)
        for wp_name, target_pd in workpoints.items():
            idx = int(np.argmin(np.abs(pd - target_pd)))
            fa_by_wp[wp_name].append(float(fa[idx] * 100.0))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    colors = ["#1f4e79", "#4a86e8", "#d95f02", "#a33030"]

    for ax, (wp_name, vals), col in zip(axes, fa_by_wp.items(), colors):
        bp = ax.boxplot([vals], patch_artist=True, tick_labels=[wp_name], widths=0.45)
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        ax.scatter(np.ones_like(vals), vals, color="black", zorder=3, s=25)
        ax.set_ylabel("Falso Alarme - FA [%]")
        ax.set_title(f"{wp_name}\nMédia FA: {np.mean(vals):.2f}%", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_roc_duel_fp32_vs_kkt(
    y_val: np.ndarray,
    y_pred_fp32: np.ndarray,
    sp_fp32: float,
    pd_fp32: float,
    fa_fp32: float,
    y_pred_q: np.ndarray,
    sp_q: float,
    pd_q: float,
    fa_q: float,
    bops_ratio: float,
    save_path: str,
    title_prefix: str = "QCNN1D KKT",
    show: bool = False,
) -> str:
    """Plots side-by-side Global ROC and Operational Trigger Zoom ROC comparing FP32 vs. Lagrange-KKT QKeras."""
    set_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    fa_f, pd_f, _ = roc_curve(y_val, y_pred_fp32)
    fa_qk, pd_qk, _ = roc_curve(y_val, y_pred_q)

    # Panel 1: Global Operational ROC
    axes[0].plot(fa_f * 100, pd_f * 100, label=f"Referência FP32 (SP={sp_fp32*100:.2f}%)", color="#1f4e79", lw=2.2)
    axes[0].plot(
        fa_qk * 100, pd_qk * 100,
        label=f"{title_prefix} ({bops_ratio*100:.1f}% BOPs, SP={sp_q*100:.2f}%)",
        color="#d95f02", lw=2.2, linestyle="--"
    )
    axes[0].plot([fa_fp32 * 100], [pd_fp32 * 100], marker="*", markersize=12, color="navy", label=f"Knee FP32 (PD={pd_fp32*100:.2f}%, FA={fa_fp32*100:.2f}%)")
    axes[0].plot([fa_q * 100], [pd_q * 100], marker="o", markersize=8, color="darkorange", label=f"Knee KKT (PD={pd_q*100:.2f}%, FA={fa_q*100:.2f}%)")

    axes[0].set_title("Duelo ROC: Referência FP32 vs. QKeras Lagrange/KKT", fontweight="bold")
    axes[0].set_xlabel("Taxa de Falso Alarme - FA [%]")
    axes[0].set_ylabel("Probabilidade de Detecção - PD [%]")
    axes[0].set_xlim(-0.2, 12.0)
    axes[0].set_ylim(88.0, 100.2)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right")

    # Panel 2: Zoom on High-Efficiency ATLAS Trigger Working Points (PD in [94%, 100%])
    axes[1].plot(fa_f * 100, pd_f * 100, label="Referência FP32", color="#1f4e79", lw=2.2)
    axes[1].plot(fa_qk * 100, pd_qk * 100, label=f"{title_prefix} (Lagrange/KKT)", color="#d95f02", lw=2.2, linestyle="--")

    wps = [("Tight (96%)", 0.96, "o", "#1b9e77"), ("Medium (97.5%)", 0.975, "s", "#7570b3"), ("Loose (98.5%)", 0.985, "^", "#e7298a")]
    for wp_label, target_pd, marker, col in wps:
        idx_q = int(np.argmin(np.abs(pd_qk - target_pd)))
        idx_f = int(np.argmin(np.abs(pd_f - target_pd)))
        axes[1].plot(
            [fa_qk[idx_q] * 100], [pd_qk[idx_q] * 100],
            marker=marker, markersize=8, color=col,
            label=f"{wp_label}: FA KKT={fa_qk[idx_q]*100:.2f}% (FP32={fa_f[idx_f]*100:.2f}%)"
        )

    axes[1].set_title("Zoom nos Pontos de Operação do Trigger ATLAS (L0 Ringer)", fontweight="bold")
    axes[1].set_xlabel("Taxa de Falso Alarme - FA [%]")
    axes[1].set_ylabel("Probabilidade de Detecção - PD [%]")
    axes[1].set_xlim(-0.1, 8.0)
    axes[1].set_ylim(94.0, 100.1)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


# Categorical slots (validated reference palette, light mode): fixed order, never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RESOURCE_LABELS = {"DSP": "DSP48E2", "LUT": "CLB LUTs", "FF": "CLB flip-flops", "BRAM_18K": "RAMB18"}


def _clean_axes(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)


def _save(fig, save_path: str, show: bool) -> str:
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_vu13p_resource_utilization(
    hw_ref: Dict[str, Any],
    hw_opt: Dict[str, Any],
    vu13p_budget: Dict[str, float],
    save_path: str,
    labels: Tuple[str, str] = ("Uniforme 8 bits", "Lagrange/KKT"),
    title_prefix: str = "Virtex UltraScale+ VU13P",
    show: bool = False,
) -> str:
    """Budget occupancy (calibrated cost model) of a reference and an optimized configuration, plus LUTs per layer."""
    set_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    keys = [k for k in ("DSP", "LUT", "FF") if k in vu13p_budget]
    x = np.arange(len(keys))
    width = 0.36
    for j, (hw, lab) in enumerate(zip((hw_ref, hw_opt), labels)):
        occ = [hw["vu13p_resources"][k] / max(vu13p_budget[k], 1.0) * 100.0 for k in keys]
        bars = axes[0].bar(x + (j - 0.5) * width, occ, width - 0.02, label=lab, color=SERIES_COLORS[j])
        for bar, v in zip(bars, occ):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    axes[0].axhline(100.0, color=INK2, linestyle="--", lw=1.2, label="Orçamento B_j (100%)")
    axes[0].set_xticks(x, [RESOURCE_LABELS[k] for k in keys])
    axes[0].set_ylabel("Ocupação do orçamento da instância [%]")
    axes[0].set_title(f"{title_prefix}: uso estimado (modelo calibrado) vs. B_j")
    axes[0].legend(frameon=False)
    _clean_axes(axes[0])

    names = [b["layer_name"] for b in hw_opt["breakdown"]]
    xl = np.arange(len(names))
    for j, (hw, lab) in enumerate(zip((hw_ref, hw_opt), labels)):
        luts = [b["LUT"] for b in hw["breakdown"]]
        axes[1].bar(xl + (j - 0.5) * width, luts, width - 0.02, label=lab, color=SERIES_COLORS[j])
    ticks = [f"{b['layer_name']}\n(w={b['b_w']}b, a_in={b['b_a_in']}b)" for b in hw_opt["breakdown"]]
    axes[1].set_xticks(xl, ticks)
    axes[1].set_ylabel("CLB LUTs (parte atribuível à camada)")
    axes[1].set_title("LUTs por camada e precisão da configuração otimizada")
    axes[1].legend(frameon=False)
    _clean_axes(axes[1])
    return _save(fig, save_path, show)


def plot_primal_dual_kkt_evolution(
    kkt_history: List[Dict[str, Any]],
    save_path: str,
    exact_optimum: Optional[float] = None,
    title_prefix: str = "Virtex UltraScale+ VU13P",
    show: bool = False,
) -> str:
    """Dual decomposition trajectory: bounds on Omega*, shadow prices mu_j and constraints g_j(x(mu_t))."""
    set_academic_style()
    it = [h["iteration"] for h in kkt_history]
    res_keys = list(kkt_history[0]["mu"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.plot(it, [h["upper_bound"] for h in kkt_history], color=SERIES_COLORS[0], lw=2, label="Melhor primal viável (UB)")
    ax.plot(it, [h["lower_bound"] for h in kkt_history], color=SERIES_COLORS[1], lw=2, label="Limite dual max q(μ) (LB)")
    if exact_optimum is not None:
        ax.axhline(exact_optimum, color=INK2, linestyle="--", lw=1.2, label="Ótimo exato (enumeração)")
    ax.set_title(f"{title_prefix}: limites de Ω*")
    ax.set_xlabel("Iteração t")
    ax.set_ylabel("Ω (aumento de perda previsto)")
    ax.legend(frameon=False)
    _clean_axes(ax)

    for j, rk in enumerate(res_keys):
        axes[1].plot(it, [h["mu"][rk] for h in kkt_history], color=SERIES_COLORS[j], lw=2, label=f"μ_{rk}")
        axes[2].plot(it, [100.0 * h["g"][rk] for h in kkt_history], color=SERIES_COLORS[j], lw=1.5, label=f"g_{rk}")
    axes[1].set_title("Preços-sombra μ_j ≥ 0")
    axes[1].set_xlabel("Iteração t")
    axes[1].set_ylabel("μ_j [Ω por unidade de B_j]")
    axes[1].legend(frameon=False)
    _clean_axes(axes[1])
    axes[2].axhline(0.0, color=INK2, linestyle="--", lw=1.2)
    axes[2].set_yscale("symlog", linthresh=10.0)  # early DSP excursions reach >1000 %; keep |g| < 10 % linear
    axes[2].set_title("Restrições g_j(x(μ_t)) (≤ 0 é viável)")
    axes[2].set_xlabel("Iteração t")
    axes[2].set_ylabel("(R_j − B_j)/B_j [%] (escala symlog)")
    axes[2].legend(frameon=False)
    _clean_axes(axes[2])
    return _save(fig, save_path, show)


def plot_sensitivity_validation(
    omega_hessian: Dict[str, Dict[int, float]],
    measured: Dict[str, Dict[int, float]],
    save_path: str,
    noise_floor: float = 1e-5,
    show: bool = False,
) -> str:
    """Hessian (2nd-order Taylor) prediction vs measured loss increase, one panel per quantized tensor."""
    set_academic_style()
    names = list(omega_hessian)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.8), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        bits = sorted(omega_hessian[name])
        ax.axhspan(1e-9, noise_floor, color=GRID, alpha=0.6, lw=0)
        ax.plot(bits, [max(omega_hessian[name][b], 1e-9) for b in bits], marker="o", color=SERIES_COLORS[0],
                lw=2, label="Ω Hessiana (Taylor)")
        ax.plot(bits, [max(measured[name][b], 1e-9) for b in bits], marker="s", color=SERIES_COLORS[1],
                lw=2, label="ΔL medido (PTQ)")
        ax.set_yscale("log")
        ax.set_xticks([b for b in bits if b in (2, 4, 6, 8, 12, 16)])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("bits")
        _clean_axes(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("aumento da perda BCE")
    first.legend(frameon=False, fontsize=9)
    first.text(0.03, 0.03, f"faixa cinza: ΔL < {noise_floor:.0e} (ruído; ≤ 0 desenhado em 1e-9)",
               transform=first.transAxes, fontsize=7.5, color=INK2)
    return _save(fig, save_path, show)


def plot_pareto_frontier(
    frontier: pd.DataFrame,
    uniform: pd.DataFrame,
    lut_budget: float,
    save_path: str,
    fp32_sp: Optional[float] = None,
    budget_label: str = "B_LUT da instância",
    show: bool = False,
) -> str:
    """Validation SP after QAT vs estimated LUTs: mixed-precision frontier (per objective) vs uniform bits."""
    set_academic_style()
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for j, (obj, grp) in enumerate(frontier.groupby("objective", sort=False)):
        grp = grp.sort_values("LUT")
        ax.plot(grp["LUT"], 100 * grp["sp_val"], marker="o", lw=2, color=SERIES_COLORS[j], label=f"Precisão mista ({obj})")
    u = uniform.sort_values("LUT")
    ax.plot(u["LUT"], 100 * u["sp_val"], marker="s", lw=2, color=SERIES_COLORS[2], label="Uniforme (w = a)")
    for _, r in u.iterrows():
        ax.annotate(f"u{int(r['bits'])}", (r["LUT"], 100 * r["sp_val"]), textcoords="offset points", xytext=(4, -12),
                    fontsize=8, color=INK2)
    ax.axvline(lut_budget, color=INK2, linestyle="--", lw=1.2, label=budget_label)
    if fp32_sp is not None:
        ax.axhline(100 * fp32_sp, color=INK2, linestyle=":", lw=1.2, label="FP32 (referência)")
    ax.set_xscale("log")
    ax.set_xlabel("LUTs estimadas na VU13P (modelo calibrado, pesos após QAT)")
    ax.set_ylabel("SP de validação após QAT [%]")
    ax.set_title("Fronteira SP × custo: alocação ótima vs. quantização uniforme")
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    return _save(fig, save_path, show)
