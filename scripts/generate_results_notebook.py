"""Generates notebooks/02_results_and_visualizations.ipynb for post-processing Jubiaba artifacts."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell(r"""# Pós-Processamento, Auditoria e Visualização dos Artefatos (Padrão Jubiaba)
## Análise Estatística e Comparativo de Hardware/Física para a Topologia `v2` (25 Anéis)

Este notebook atua como o equivalente ao `Results_CNN_L0Ringer.py` do `codeTCC`, operando de forma **100% desacoplada do treinamento**. Ele lê diretamente os artefatos estruturados no padrão **Jubiaba**:

```text
results/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/
├── saved_models/  # model_v{V}_f{F}_r{R}.keras, best_model.keras, best_model.onnx
├── plots/         # Gráficos em alta resolução (300 DPI)
├── history/       # training_history.csv, history.json
├── metadata/      # training_metadata_v{V}.pkl/.json, cross_validation_results.csv, reproducibility_metadata.json, kkt_allocation_results.json
└── evaluation/    # holdout_test_set.npz, holdout_predictions.npz
```
"""))

# Cell 1: Imports & Path Configuration
cells.append(nbf.v4.new_code_cell(r"""import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if not os.path.exists(os.path.join(repo_root, 'src')):
    repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Seleção do Experimento a Inspecionar
EXPORT_DIR = os.path.join(repo_root, "results")
DATASET_PREFIX = "mc21_13p6TeV.Zee.JF17.2sigma.5M"
MODEL_NAME = "QCNN1D_KKT"
IET = 5
IETA = 0
VERSION = 2  # 2 -> 25 Anéis, 1 -> 50 Anéis, 0 -> 100 Anéis

base_dir = os.path.join(EXPORT_DIR, DATASET_PREFIX, MODEL_NAME, f"iet{IET}", f"ieta{IETA}", f"v{VERSION}")
print(f"Inspecionando diretório Jubiaba:\n  -> {base_dir}")
assert os.path.isdir(base_dir), f"Diretório não encontrado: {base_dir}"
"""))

# Cell 2: Inspect Reproducibility & KKT Allocation Metadata
cells.append(nbf.v4.new_markdown_cell(r"""## 1. Metadados de Reprodutibilidade e Alocação Analítica de Bits (KKT)

Carregamos `reproducibility_metadata.json` e `kkt_allocation_results.json` da pasta `metadata/`.
"""))

cells.append(nbf.v4.new_code_cell(r"""repro_path = os.path.join(base_dir, "metadata", "reproducibility_metadata.json")
kkt_path = os.path.join(base_dir, "metadata", "kkt_allocation_results.json")

with open(repro_path, "r", encoding="utf-8") as f:
    repro_meta = json.load(f)

with open(kkt_path, "r", encoding="utf-8") as f:
    kkt_meta = json.load(f)

print("=== Metadados de Reprodutibilidade ===")
print(f"  • Timestamp       : {repro_meta['timestamp']}")
print(f"  • Git Commit      : {repro_meta['git_commit']}")
print(f"  • Região Cinemática: IET={repro_meta['iet']}, IETA={repro_meta['ieta']} | Versão: v{repro_meta['version']}")
print(f"  • Estrutura Anéis : {repro_meta['extra_config'].get('rings_structure')}")

kkt_rows = []
hw_q = repro_meta["extra_config"]["hardware_qkeras_kkt"]
for layer_info in hw_q["breakdown"]:
    lname = layer_info["layer_name"]
    kkt_rows.append({
        "Camada": lname,
        "Tipo": layer_info["layer_type"],
        "Parâmetros": layer_info["params"],
        "MACs": layer_info["macs"],
        "Bits Contínuos (KKT)": round(kkt_meta["continuous_bits"][lname], 3),
        "Bits Inteiros (QKeras)": kkt_meta["integer_bits"][lname],
        "BOPs da Camada": layer_info["bops"],
    })

df_kkt = pd.DataFrame(kkt_rows)
print("\n=== Tabela de Alocação de Bits por Camada (KKT) ===")
display(df_kkt)
"""))

# Cell 3: Cross-Validation Results Table
cells.append(nbf.v4.new_markdown_cell(r"""## 2. Desempenho Estatístico em Validação Cruzada ($K$-Folds)

Analisamos os resultados salvos em `metadata/cross_validation_results.csv`.
"""))

cells.append(nbf.v4.new_code_cell(r"""cv_csv_path = os.path.join(base_dir, "metadata", "cross_validation_results.csv")
df_cv = pd.read_csv(cv_csv_path)

df_cv_display = df_cv.copy()
for col in ["sp_max", "pd_knee", "fa_knee", "auc"]:
    df_cv_display[col] = (df_cv_display[col] * 100.0).round(2)

display(df_cv_display)

summary_stats = pd.DataFrame({
    "Métrica": ["Índice SP [%]", "Prob. Detecção - PD [%]", "Falso Alarme - FA [%]", "AUC ROC [%]"],
    "Média": [
        df_cv["sp_max"].mean() * 100,
        df_cv["pd_knee"].mean() * 100,
        df_cv["fa_knee"].mean() * 100,
        df_cv["auc"].mean() * 100,
    ],
    "Desvio Padrão": [
        df_cv["sp_max"].std() * 100,
        df_cv["pd_knee"].std() * 100,
        df_cv["fa_knee"].std() * 100,
        df_cv["auc"].std() * 100,
    ],
    "Melhor Fold": [
        df_cv["sp_max"].max() * 100,
        df_cv.loc[df_cv["sp_max"].idxmax(), "pd_knee"] * 100,
        df_cv.loc[df_cv["sp_max"].idxmax(), "fa_knee"] * 100,
        df_cv["auc"].max() * 100,
    ],
}).round(2)

print("\n=== Resumo Estatístico dos Folds ===")
display(summary_stats)
"""))

# Cell 4: Holdout Verification from .npz
cells.append(nbf.v4.new_markdown_cell(r"""## 3. Auditoria Independente a partir de `evaluation/holdout_predictions.npz`

Verificamos as predições brutas armazenadas em disco nos pontos de operação padrão do trigger ATLAS (`Tight`, `Medium`, `Loose`, `Very Loose`).
"""))

cells.append(nbf.v4.new_code_cell(r"""from sklearn.metrics import roc_curve
from src.metrics import compute_sp_with_threshold

preds_npz = np.load(os.path.join(base_dir, "evaluation", "holdout_predictions.npz"))
y_true_ho = preds_npz["y_true"]
y_pred_ho = preds_npz["y_pred"]
saved_thresh = float(preds_npz["threshold"][0])

sp_ho, pd_ho, fa_ho, thresh_ho, _ = compute_sp_with_threshold(y_true_ho, y_pred_ho)
fa_curve, pd_curve, thresholds = roc_curve(y_true_ho, y_pred_ho)

wp_targets = [
    ("Knee SP (Ótimo)", pd_ho),
    ("Tight (PD = 96.0%)", 0.960),
    ("Medium (PD = 97.5%)", 0.975),
    ("Loose (PD = 98.5%)", 0.985),
    ("Very Loose (PD = 99.0%)", 0.990),
]

wp_rows = []
for wp_name, target_pd in wp_targets:
    idx = int(np.argmin(np.abs(pd_curve - target_pd)))
    wp_rows.append({
        "Ponto de Operação": wp_name,
        "PD Alvo [%]": round(target_pd * 100, 2),
        "PD Realizado [%]": round(float(pd_curve[idx]) * 100, 2),
        "Falso Alarme - FA [%]": round(float(fa_curve[idx]) * 100, 2),
        "Rejeição de Jatos (1-FA) [%]": round((1.0 - float(fa_curve[idx])) * 100, 2),
        "Limiar de Decisão": round(float(thresholds[idx]), 4),
    })

df_wp = pd.DataFrame(wp_rows)
display(df_wp)
"""))

# Cell 5: Gallery of Plots from plots/
cells.append(nbf.v4.new_markdown_cell(r"""## 4. Galeria Completa de Visualizações (`plots/`)

Exibimos todas as figuras científicas geradas na pasta `plots/` do experimento.
"""))

cells.append(nbf.v4.new_code_cell(r"""plots_dir = os.path.join(base_dir, "plots")
plot_files = [
    f"rings_profile_v{VERSION}.png",
    "learning_curves.png",
    "roc_best_model.png",
    "roc_cv_envelope.png",
    "output_scores_hist.png",
    "confusion_matrix_knee.png",
    "cv_boxplots.png",
    "fa_per_fold_workpoints.png",
    "pareto_and_roc_duel.png",
]

for pfile in plot_files:
    full_p = os.path.join(plots_dir, pfile)
    if os.path.exists(full_p):
        print(f"\n--- {pfile} ---")
        display(Image(filename=full_p, width=900))
"""))

nb["cells"] = cells
out_path = "notebooks/02_results_and_visualizations.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook de visualizações gerado em: {out_path}")
