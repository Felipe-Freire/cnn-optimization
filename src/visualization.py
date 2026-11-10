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


def plot_vu13p_resource_utilization(
    hw_8bit: Dict[str, Any],
    hw_kkt: Dict[str, Any],
    vu13p_budget: Dict[str, float],
    save_path: str,
    title_prefix: str = "Virtex UltraScale+ VU13P",
    show: bool = False,
) -> str:
    """Plots VU13P resource consumption (% of instance budget) and per-layer LUT/BRAM breakdown."""
    set_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # Panel 1: Resource Utilization (% of Instance Budget B_j) for LUT, FF, BRAM_bits
    res_keys = ["LUT", "FF", "BRAM_bits"]
    res_labels = ["CLB LUTs", "CLB Flip-Flops", "Block RAM [bits]"]

    u_8b = [hw_8bit["vu13p_resources"][k] / max(vu13p_budget[k], 1.0) * 100.0 for k in res_keys]
    u_kkt = [hw_kkt["vu13p_resources"][k] / max(vu13p_budget[k], 1.0) * 100.0 for k in res_keys]

    x = np.arange(len(res_keys))
    width = 0.35

    b1 = axes[0].bar(x - width / 2, u_8b, width, label="Uniforme 8-bit (Inviável em LUT/BRAM)", color="#a33030", alpha=0.8)
    b2 = axes[0].bar(x + width / 2, u_kkt, width, label="Otimizado Lagrange/KKT (Viável ≤ 100%)", color="#1b9e77", alpha=0.85)
    axes[0].axhline(100.0, color="black", linestyle="--", lw=1.8, label="Limite de Orçamento VU13P (100%)")

    for bar in b1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(res_labels, fontweight="bold")
    axes[0].set_ylabel("Ocupação do Orçamento da Instância [%]")
    axes[0].set_title(f"{title_prefix}: Uso de Recursos vs. Limites B_j (DSP = 0)", fontweight="bold")
    axes[0].set_ylim(0, max(max(u_8b), 115.0) * 1.15)
    axes[0].grid(True, linestyle=":", alpha=0.5, axis="y")
    axes[0].legend(loc="upper right")

    # Panel 2: Per-Layer LUT & Bit Allocation Breakdown in KKT Model
    layers = [item["layer_name"] for item in hw_kkt["breakdown"]]
    luts_8b = [item["LUT"] for item in hw_8bit["breakdown"]]
    luts_kkt = [item["LUT"] for item in hw_kkt["breakdown"]]
    bit_labels = [f"{item['layer_name']}\n(w={item['b_w']}b, a={item['b_a']}b)" for item in hw_kkt["breakdown"]]

    x_l = np.arange(len(layers))
    b3 = axes[1].bar(x_l - width / 2, luts_8b, width, label="Uniforme (w=8b, a=8b)", color="#1f4e79", alpha=0.75)
    b4 = axes[1].bar(x_l + width / 2, luts_kkt, width, label="Alocação KKT (w_l*, a_l*)", color="#d95f02", alpha=0.85)

    for bar in b3:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 30.0, f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    for bar in b4:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 30.0, f"{h:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[1].set_xticks(x_l)
    axes[1].set_xticklabels(bit_labels, fontweight="bold")
    axes[1].set_ylabel("CLB LUTs Consumidas")
    axes[1].set_title("Consumo de LUTs e Precisão (w_l, a_l) por Camada", fontweight="bold")
    axes[1].set_ylim(0, max(luts_8b) * 1.18)
    axes[1].grid(True, linestyle=":", alpha=0.5, axis="y")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_primal_dual_kkt_evolution(
    kkt_history: List[Dict[str, Any]],
    save_path: str,
    title_prefix: str = "Virtex UltraScale+ VU13P",
    show: bool = False,
) -> str:
    """Plots Primal-Dual Lagrange-KKT optimization trajectory (SP-Index, Shadow Prices mu_j, and Constraints g_j)."""
    set_academic_style()
    iters = [item["iteration"] for item in kkt_history]
    sp_vals = [item["sp_val"] * 100.0 for item in kkt_history]
    lag_costs = [item["lagrangian_cost"] for item in kkt_history]
    conv_iters = [item["iteration"] for item in kkt_history if item.get("kkt_diagnostics", {}).get("converged", False)]

    res_keys = list(kkt_history[0]["mu"].keys())
    colors = {"DSP": "#e7298a", "LUT": "#1f4e79", "FF": "#1b9e77", "BRAM_bits": "#d95f02"}
    markers = {"DSP": "^", "LUT": "o", "FF": "s", "BRAM_bits": "D"}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # 1. SP-Index & Lagrangian Objective Evolution
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(iters, sp_vals, color="#1f4e79", marker="o", lw=2.2, label="Índice-SP Val [%]")
    l2 = ax1_twin.plot(
        iters, lag_costs, color="#d95f02", marker="s", linestyle="--", lw=1.8, label="Custo Lagrangeano L(x, μ)"
    )
    if conv_iters:
        ax1.axvline(conv_iters[0], color="forestgreen", linestyle=":", lw=2.0, label=f"Convergência KKT (t={conv_iters[0]})")
    ax1.set_title(f"{title_prefix}: Índice-SP vs. Iteração", fontweight="bold")
    ax1.set_xlabel("Iteração Primal-Dual (t)")
    ax1.set_ylabel("Índice-SP de Validação [%]", color="#1f4e79")
    ax1_twin.set_ylabel("Custo Lagrangeano L(x, μ)", color="#d95f02")
    ax1.grid(True, linestyle=":", alpha=0.5)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    # 2. Evolution of Lagrange Multipliers (Shadow Prices mu_j)
    ax2 = axes[1]
    for rk in res_keys:
        mu_series = [item["mu"][rk] for item in kkt_history]
        ax2.plot(
            iters, mu_series,
            marker=markers.get(rk, "o"),
            color=colors.get(rk, "black"),
            lw=2.0,
            label=f"μ_{rk} (Preço de Sombra)",
        )
    if conv_iters:
        ax2.axvline(conv_iters[0], color="forestgreen", linestyle=":", lw=2.0, label=f"Ótimo KKT (t={conv_iters[0]})")
    ax2.set_title("Evolução dos Multiplicadores (μ_j) - VU13P", fontweight="bold")
    ax2.set_xlabel("Iteração Primal-Dual (t)")
    ax2.set_ylabel("Multiplicador Dual (μ_j ≥ 0)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="best")

    # 3. Normalized Constraint Violations g_j(x) <= 0
    ax3 = axes[2]
    ax3.axhline(0.0, color="crimson", linestyle="--", lw=1.8, label="Limite Viável (g_j = 0)")
    for rk in res_keys:
        g_series = [item["g_val"][rk] * 100.0 for item in kkt_history]
        ax3.plot(
            iters, g_series,
            marker=markers.get(rk, "o"),
            color=colors.get(rk, "black"),
            lw=1.8,
            label=f"g_{rk}(x) [%]",
        )
    if conv_iters:
        ax3.axvline(conv_iters[0], color="forestgreen", linestyle=":", lw=2.0)
    ax3.set_title("Viabilidade Primal das Restrições g_j(x) ≤ 0", fontweight="bold")
    ax3.set_xlabel("Iteração Primal-Dual (t)")
    ax3.set_ylabel("Excesso / Folga Relativa ao Orçamento [%]")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path
