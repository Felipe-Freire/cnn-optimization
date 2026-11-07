"""Generates the comprehensive research notebook for QKeras CNN1D optimization."""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell(r"""# Otimização de CNN1D Quantizada para FPGA (ATLAS L0 Ringer)
## Alocação de Bits por Lagrange/KKT e Fronteira de Pareto Multi-Objetivo (NSGA-II)

Este notebook implementa a metodologia de codesign de rede neural compacta para o sistema de filtragem de elétrons do ATLAS L0 Ringer.

### Objetivos do Experimento:
1. **Redução Arquitetural**: Simplificar a CNN1D original de 2 convoluções e 2 densas para uma estrutura compacta viável em hardware FPGA;
2. **Quantização Heterogênea Ótima**: Determinar a precisão ótima de bits de cada camada via expansão de Taylor de 2ª ordem da perda $\Delta \mathcal{L}$, estimativa do traço da Hessiana via algoritmo de Hutchinson, e solução analítica das condições KKT (*water-filling*);
3. **Mapeamento de Hiperparâmetros Bi-Objetivo**: Gerar a **Fronteira de Pareto** ($\max SP$ vs. $\min \text{BOPs}$) via algoritmo genético NSGA-II e busca TPE com penalidade de hardware.
"""))

# Cell 1: Environment & Imports
cells.append(nbf.v4.new_code_cell(r"""import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras
import qkeras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

# Adicionar raiz do repositório ao PYTHONPATH
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if not os.path.exists(os.path.join(repo_root, 'src')):
    repo_root = os.path.abspath(os.getcwd())

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.metrics import compute_sp, norm1, SPCallback
from src.hardware import compute_model_hardware_costs
from src.kkt_allocator import (
    compute_hutchinson_hessian_trace,
    compute_layer_c_coefficients,
    solve_kkt_bit_allocation
)
from src.models import build_compact_cnn1d_fp32, build_compact_qcnn1d

print("TensorFlow Version:", tf.__version__)
print("QKeras Version:", qkeras.__version__)
print("Repository Root:", repo_root)
"""))

# Cell 2: Data Ingestion & Preprocessing
cells.append(nbf.v4.new_markdown_cell(r"""## 1. Ingestão e Pré-processamento dos Dados Ringer

Os anéis de energia do calorímetro representam a deposição transversal em coordenadas concêntricas.
Aplicamos a normalização L1 por evento:

$$\hat{x}_i = \frac{x_i}{\sum_{j=1}^M |x_j|}$$

E fatiamos para a resolução reduzida $v_1$ (50 anéis).
"""))

cells.append(nbf.v4.new_code_cell(r"""# Parâmetros da Região do Detector
iet = 5
ieta = 0
data_path = os.path.join(repo_root, 'data', f'mc21_13p6TeV.Zee.JF17.2sigma.5M.et{iet}_eta{ieta}.h5')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Arquivo de dados não encontrado em: {data_path}")

print(f"Carregando dados de: {data_path}")
df_loaded = pd.read_hdf(data_path, key='data')

ring_columns = [col for col in df_loaded.columns if col.startswith('trig_L2_cl_ring_')]
ring_columns = sorted(ring_columns, key=lambda x: int(x.split('_')[-1]))

signal_df = df_loaded[df_loaded['target'] == 1][ring_columns]
background_df = df_loaded[df_loaded['target'] == 0][ring_columns]

ns = min(len(signal_df), len(background_df))
print(f"Total de amostras balanceadas: {ns*2} (Sinal: {ns}, Fundo: {ns})")

# Normalização de energia
sig_normed = norm1(signal_df.values[:ns])
bkg_normed = norm1(background_df.values[:ns])

# Versão v1: 50 anéis (fator de redução = 2 nas camadas do calorímetro)
ESTRUTURA_BASE = {"PS": 8, "EM1": 64, "EM2": 8, "EM3": 8, "HAD1": 4, "HAD2": 4, "HAD3": 4}
fator_reducao = 2
indices_v1 = []
ponteiro = 0
for camada, qtd in ESTRUTURA_BASE.items():
    indices_v1.extend(range(ponteiro, ponteiro + qtd // fator_reducao))
    ponteiro += qtd

X_v1 = np.vstack([sig_normed[:, indices_v1], bkg_normed[:, indices_v1]]).astype(np.float32)
y = np.vstack([np.ones((ns, 1)), np.zeros((ns, 1))]).ravel().astype(np.float32)

print(f"Shape final do dataset X_v1: {X_v1.shape}, Labels y: {y.shape}")
"""))

# Cell 3: Baseline FP32 Model & Training
cells.append(nbf.v4.new_markdown_cell(r"""## 2. Treinamento da CNN1D Compacta em FP32 (Baseline)

Treinamos uma CNN1D enxuta (1 camada de convolução com 8 filtros, MaxPooling e 1 camada densa de 8 neurônios) monitorando o **Índice Soma-Produto ($SP$)** na validação:

$$SP = \sqrt{\sqrt{PD \cdot (1 - FA)} \cdot \left[ \frac{PD + (1 - FA)}{2} \right]}$$
"""))

cells.append(nbf.v4.new_code_cell(r"""# Divisão estratificada Treino / Validação
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_v1, y))

X_train, y_train = X_v1[train_idx], y[train_idx]
X_val, y_val = X_v1[val_idx], y[val_idx]

input_dim = X_v1.shape[1]
model_fp32 = build_compact_cnn1d_fp32(input_dim=input_dim, num_filters=8, kernel_size=3, dense_units=8)

sp_cb_fp32 = SPCallback(val_data=(X_val, y_val), patience=10, verbose=True)

history_fp32 = model_fp32.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=512,
    callbacks=[sp_cb_fp32],
    verbose=1
)

sp_cb_fp32.restore_best_weights()
hw_fp32 = compute_model_hardware_costs(model_fp32, input_dim=input_dim)

print(f"\n[Baseline FP32] Melhor SP: {sp_cb_fp32.best_sp:.4f} (PD: {sp_cb_fp32.best_pd_at_knee:.4f}, FA: {sp_cb_fp32.best_fa_at_knee:.4f})")
print(f"[Hardware Baseline] Total MACs: {hw_fp32['total_macs']}, Parâmetros: {hw_fp32['total_params']}")
"""))

# Cell 4: Hessian Trace & KKT Bit Allocation
cells.append(nbf.v4.new_markdown_cell(r"""## 3. Alocação Ótima de Bits por Multiplicadores de Lagrange e Condições KKT

A perda esperada sob quantização de pesos com ruído $\sigma_i^2 \propto 4^{-b_i}$ é:

$$\min_{\mathbf{b}} \sum_{i=1}^L c_i 4^{-b_i} \quad \text{s.a.} \quad \sum_{i=1}^L a_i b_i \le B, \quad b_{\min} \le b_i \le b_{\max}$$

onde:
- $c_i = \operatorname{Tr}(\mathbf{H}_i) \frac{n_i r_i^2}{3}$ é o fator de sensibilidade intrínseca da camada $i$;
- $a_i = \text{MAC}_i \cdot b_{a,i}$ é o custo em hardware por bit;
- A solução analítica KKT (*water-filling*) é dada por:

$$b_i^*(\lambda) = \min\left( b_{\max}, \max\left( b_{\min}, \frac{1}{2} \log_2 \left( \frac{c_i \ln 4}{\lambda a_i} \right) \right) \right)$$
"""))

cells.append(nbf.v4.new_code_cell(r"""# 1. Estimativa do Traço da Hessiana via Algoritmo de Hutchinson (Monte Carlo com Rademacher)
sample_idx = np.random.choice(len(X_val), size=512, replace=False)
x_sample_tf = tf.convert_to_tensor(X_val[sample_idx])
y_sample_tf = tf.convert_to_tensor(y_val[sample_idx], dtype=tf.float32)[:, None]
bce_loss = keras.losses.BinaryCrossentropy()

print("Calculando traço da Hessiana Tr(H_i) via produtos Hessiana-vetor...")
traces = compute_hutchinson_hessian_trace(model_fp32, bce_loss, x_sample_tf, y_sample_tf, num_hvp=10)
c_coeffs = compute_layer_c_coefficients(model_fp32, traces)

# 2. Extração dos custos de hardware a_i = MACs_i * b_a,i
layer_names = [item['layer_name'] for item in hw_fp32['breakdown']]
a_costs = [item['macs'] * item['b_a'] for item in hw_fp32['breakdown']]
c_list = [c_coeffs[name] for name in layer_names]

# 3. Definição de Orçamentos de Hardware (BOPs)
# 8 bits uniforme equivale ao orçamento máximo
bops_uniform_8bit = sum(a * 8 for a in a_costs)
budget_target = bops_uniform_8bit * 0.60  # Redução de 40% no consumo de BOPs

kkt_result = solve_kkt_bit_allocation(
    layer_names=layer_names,
    c_coeffs=c_list,
    a_costs=a_costs,
    budget_bops=budget_target,
    b_min=2,
    b_max=8
)

print(f"\nOrçamento Máximo (8-bit uniforme): {bops_uniform_8bit:,.0f} BOPs")
print(f"Orçamento Alvo (60%): {budget_target:,.0f} BOPs")
print(f"Preço-Sombra Ótimo (λ*): {kkt_result['lambda_optimal']:.6e}")
print("\n=== Alocação Ótima de Bits por Camada (KKT) ===")
for name in layer_names:
    b_cont = kkt_result['continuous_bits'][name]
    b_int = kkt_result['integer_bits'][name]
    print(f"• {name:15s} | Tr(H): {traces[name]:.4e} | Custo a_i: {a_costs[layer_names.index(name)]:6.0f} | Bits Contínuos: {b_cont:.2f} | Bits Inteiros: {b_int}b")

print(f"\nBOPs Realizados pelo Modelo KKT: {kkt_result['achieved_bops']:,.0f} ({kkt_result['achieved_bops']/bops_uniform_8bit*100:.1f}% do 8-bit)")
"""))

# Cell 5: QKeras Training with KKT Bits
cells.append(nbf.v4.new_markdown_cell(r"""## 4. Treinamento da CNN1D Quantizada com Precisão Mista (KKT QAT)

Instanciamos o modelo QKeras com a alocação de bits determinada matematicamente por KKT e executamos o treinamento ciente de quantização (*Quantization-Aware Training* - QAT).
"""))

cells.append(nbf.v4.new_code_cell(r"""kkt_weights_bits = kkt_result['integer_bits']

# Modelo QKeras com alocação KKT
model_qkeras_kkt = build_compact_qcnn1d(
    input_dim=input_dim,
    num_filters=8,
    kernel_size=3,
    dense_units=8,
    weights_bits=kkt_weights_bits,
    act_bits=8
)

sp_cb_qkeras = SPCallback(val_data=(X_val, y_val), patience=10, verbose=True)

history_qkeras = model_qkeras_kkt.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=512,
    callbacks=[sp_cb_qkeras],
    verbose=1
)

sp_cb_qkeras.restore_best_weights()
hw_qkeras_kkt = compute_model_hardware_costs(model_qkeras_kkt, weights_bits=kkt_weights_bits, input_dim=input_dim)

print(f"\n[QKeras KKT Mixed-Precision] Melhor SP: {sp_cb_qkeras.best_sp:.4f} (PD: {sp_cb_qkeras.best_pd_at_knee:.4f}, FA: {sp_cb_qkeras.best_fa_at_knee:.4f})")
print(f"[Hardware KKT] Total BOPs: {hw_qkeras_kkt['total_bops']:,} | Parâmetros: {hw_qkeras_kkt['total_params']}")
"""))

# Cell 6: Multi-Objective NSGA-II & Pareto Frontier
cells.append(nbf.v4.new_markdown_cell(r"""## 5. Otimização Bi-Objetivo e Fronteira de Pareto (NSGA-II)

Avaliamos simultaneamente os dois objetivos conflitantes do trigger:

$$\max f_1(\boldsymbol{\theta}) = SP_{\text{val}}(\boldsymbol{\theta})$$

$$\min f_2(\boldsymbol{\theta}) = \text{BOPs}(\boldsymbol{\theta})$$

Utilizamos o algoritmo **NSGA-II** (*Non-dominated Sorting Genetic Algorithm II*) via Optuna para encontrar a curva de *trade-off* exata da arquitetura.
"""))

cells.append(nbf.v4.new_code_cell(r"""import optuna

def objective(trial):
    n_filters = trial.suggest_categorical("filters", [4, 8, 16])
    k_size = trial.suggest_categorical("kernel_size", [3, 5])
    d_units = trial.suggest_categorical("dense_units", [4, 8])
    b_conv = trial.suggest_categorical("b_conv", [4, 6, 8])
    b_dense = trial.suggest_categorical("b_dense", [4, 6, 8])
    lr = trial.suggest_float("lr", 1e-3, 5e-3, log=True)

    weights_bits = {
        "conv1d_1": b_conv,
        "dense_1": b_dense,
        "output_dense": 8
    }

    model_trial = build_compact_qcnn1d(
        input_dim=input_dim,
        num_filters=n_filters,
        kernel_size=k_size,
        dense_units=d_units,
        weights_bits=weights_bits,
        act_bits=8,
        learning_rate=lr
    )

    costs = compute_model_hardware_costs(model_trial, weights_bits=weights_bits, input_dim=input_dim)
    bops = float(costs["total_bops"])

    # Treino rápido de avaliação (5 épocas)
    sp_cb_trial = SPCallback(val_data=(X_val, y_val), patience=4, verbose=False)
    model_trial.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=1024,
        callbacks=[sp_cb_trial],
        verbose=0
    )

    sp_val = float(sp_cb_trial.best_sp)
    return sp_val, bops

study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(seed=42)
)

print("Iniciando busca bi-objetivo com NSGA-II (10 trials para demonstração)...")
study.optimize(objective, n_trials=10)

print("\n=== Soluções na Fronteira de Pareto ===")
pareto_trials = study.best_trials
for t in pareto_trials:
    print(f"Trial #{t.number}: SP = {t.values[0]:.4f} | BOPs = {t.values[1]:,.0f} | Params: {t.params}")
"""))

# Cell 7: Plots & ROC Analysis
cells.append(nbf.v4.new_markdown_cell(r"""## 6. Curvas ROC e Fronteira de Pareto Visual para o Artigo

Geramos as curvas de desempenho para comparação direta e publicação.
"""))

cells.append(nbf.v4.new_code_cell(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Curvas ROC Sobrepostas
y_pred_fp32 = model_fp32.predict(X_val, batch_size=2048, verbose=0).ravel()
y_pred_qkeras = model_qkeras_kkt.predict(X_val, batch_size=2048, verbose=0).ravel()

fa_fp32, pd_fp32, _ = roc_curve(y_val, y_pred_fp32)
fa_q, pd_q, _ = roc_curve(y_val, y_pred_qkeras)

axes[0].plot(fa_fp32, pd_fp32, label=f'Baseline FP32 (SP={sp_cb_fp32.best_sp:.4f})', color='#1f4e79', lw=2)
axes[0].plot(fa_q, pd_q, label=f'KKT QKeras ({kkt_result["achieved_bops"]/bops_uniform_8bit*100:.0f}% BOPs, SP={sp_cb_qkeras.best_sp:.4f})', color='#d95f02', lw=2, linestyle='--')
axes[0].plot([sp_cb_fp32.best_fa_at_knee], [sp_cb_fp32.best_pd_at_knee], marker='*', markersize=12, color='navy', label='Knee FP32')
axes[0].plot([sp_cb_qkeras.best_fa_at_knee], [sp_cb_qkeras.best_pd_at_knee], marker='o', markersize=8, color='darkorange', label='Knee QKeras KKT')

axes[0].set_title("Curva ROC e Pontos de Operação (Knee SP)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Taxa de Falso Alarme (FA)")
axes[0].set_ylabel("Probabilidade de Detecção (PD)")
axes[0].set_xlim(-0.01, 0.4)
axes[0].set_ylim(0.6, 1.01)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='lower right')

# 2. Fronteira de Pareto: SP vs. BOPs
all_sp = [t.values[0] for t in study.trials if t.values is not None]
all_bops = [t.values[1] for t in study.trials if t.values is not None]
pareto_sp = [t.values[0] for t in pareto_trials]
pareto_bops = [t.values[1] for t in pareto_trials]

axes[1].scatter(all_bops, all_sp, color='gray', alpha=0.5, label='Espaço Explorado')
axes[1].scatter(pareto_bops, pareto_sp, color='#e7298a', s=90, edgecolor='black', label='Fronteira de Pareto (NSGA-II)')
axes[1].scatter([hw_qkeras_kkt['total_bops']], [sp_cb_qkeras.best_sp], color='green', marker='^', s=120, label='Ponto Ótimo KKT')

axes[1].set_title("Fronteira de Pareto: Desempenho vs. Custo em Silício", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Custo de Hardware (BOPs)")
axes[1].set_ylabel("Índice Soma-Produto (SP)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='lower right')

plt.tight_layout()
plots_dir = os.path.join(repo_root, 'docs', 'plots')
os.makedirs(plots_dir, exist_ok=True)
plot_save_path = os.path.join(plots_dir, 'comparativo_roc_pareto.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Gráficos salvos com sucesso em: {plot_save_path}")
"""))

# Cell 8: Summary Table for Article
cells.append(nbf.v4.new_markdown_cell(r"""## 7. Tabela Síntese para Escrita de Artigo

Tabela comparativa consolidada com métricas de física ($SP, PD, FA$) e métricas de síntese de hardware ($\text{MACs}, \text{BOPs}, \text{Bits de Pesos}$).
"""))

cells.append(nbf.v4.new_code_cell(r"""summary_data = [
    {
        "Modelo": "Baseline FP32 Compacto",
        "Precisão": "Float32",
        "SP Max": f"{sp_cb_fp32.best_sp:.4f}",
        "PD Knee": f"{sp_cb_fp32.best_pd_at_knee:.4f}",
        "FA Knee": f"{sp_cb_fp32.best_fa_at_knee:.4f}",
        "MACs": f"{hw_fp32['total_macs']:,}",
        "BOPs": f"{hw_fp32['total_bops']:,}",
        "Bits Pesos": f"{hw_fp32['total_weight_bits']:,}"
    },
    {
        "Modelo": "QKeras KKT Heterogêneo",
        "Precisão": f"Conv: {kkt_weights_bits['conv1d_1']}b | Dense: {kkt_weights_bits['dense_1']}b | Out: {kkt_weights_bits['output_dense']}b",
        "SP Max": f"{sp_cb_qkeras.best_sp:.4f}",
        "PD Knee": f"{sp_cb_qkeras.best_pd_at_knee:.4f}",
        "FA Knee": f"{sp_cb_qkeras.best_fa_at_knee:.4f}",
        "MACs": f"{hw_qkeras_kkt['total_macs']:,}",
        "BOPs": f"{hw_qkeras_kkt['total_bops']:,}",
        "Bits Pesos": f"{hw_qkeras_kkt['total_weight_bits']:,}"
    }
]

df_summary = pd.DataFrame(summary_data)
display(df_summary)
"""))

nb['cells'] = cells
notebook_path = "notebooks/01_quantized_cnn1d_optimization.ipynb"
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook gerado com sucesso em: {notebook_path}")
