"""Generates the streamlined 3-stage QKeras -> SP/VU13P -> Lagrange/KKT notebook."""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell(r"""# Otimização de CNN1D Quantizada em Ciclo Fechado (QKeras $\to$ SP/VU13P $\to$ Lagrange/KKT)
## Alocação Primal-Dual de Bits na FPGA Xilinx Virtex UltraScale+ VU13P para o ATLAS L0 Ringer (`v2` = 25 Anéis)

Este notebook implementa de forma enxuta e direta o **ciclo fechado de otimização em 3 estágios** para síntese da CNN1D compacta em hardware FPGA:

```text
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. REDE NEURAL QUANTIZADA (QKeras)                                      │
  │    Configura larguras de bits e arquitetura (w_l, a_l, n_l) + Warm-Start│
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 2. AVALIAÇÃO DE DESEMPENHO E CUSTO                                      │
  │    Calcula a Perda pelo Índice-SP (1 - SP) e estima recursos na VU13P   │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3. NÚCLEO DE OTIMIZAÇÃO LAGRANGE / KKT                                  │
  │    Compara uso com limites B_j (LUTs, DSPs, FFs, BRAM), atualiza os     │
  │    Preços de Sombra (μ_j) e recalcula (w_l*, a_l*) até convergência KKT │
  └─────────────────────────────────────────────────────────────────────────┘
```

Todos os artefatos são exportados automaticamente no padrão hierárquico **Jubiaba**:
`<export_dir>/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/` (`saved_models/`, `plots/`, `history/`, `metadata/`, `evaluation/`).
"""))

# Cell 1: Environment & Imports
cells.append(nbf.v4.new_code_cell(r"""import os
import sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Adicionar raiz do repositório ao PYTHONPATH
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if not os.path.exists(os.path.join(repo_root, 'src')):
    repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# GPU CUDA se o TensorFlow conseguir executar nela; caso contrário, CPU (antes de importar o TensorFlow)
from src.device import select_device
DEVICE = select_device()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras
import qkeras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc

from src.metrics import (
    ESTRUTURA_BASE,
    CONFIG_VERSOES,
    slice_rings_by_version,
    norm1,
    compute_sp,
    compute_sp_with_threshold,
    SPCallback,
)
from src.training import evaluate_model, paired_difference, train_fp32, train_qat
from src.hardware import (
    VU13P_CHIP_LIMITS,
    VU13P_LIMITS,
    get_cost_model,
    get_vu13p_budget,
    compute_model_hardware_costs,
    make_cost_fn,
)
from src.kkt_allocator import (
    compute_hutchinson_hessian_trace,
    compute_layer_c_coefficients,
    LagrangeKKTQuantizerOptimizer,
    check_kkt_stopping_criteria,
)
from src.models import (
    build_compact_cnn1d_fp32,
    build_compact_qcnn1d,
    transfer_weights_to_qmodel,
)
from src.exporter import ArtifactExporter
from src.visualization import (
    plot_rings_profile,
    plot_learning_curves,
    plot_best_model_roc,
    plot_cv_roc_envelope,
    plot_output_score_histogram,
    plot_confusion_matrix_knee,
    plot_cv_statistics_boxplots,
    plot_fa_per_fold_workpoints,
    plot_roc_duel_fp32_vs_kkt,
    plot_vu13p_resource_utilization,
    plot_primal_dual_kkt_evolution,
)

SEED = 42
keras.utils.set_random_seed(SEED)           # Python, NumPy e TensorFlow
tf.config.experimental.enable_op_determinism()  # mesma execução -> mesmos números

print("TensorFlow Version:", tf.__version__)
print("QKeras Version:", qkeras.__version__)
print("Dispositivo:", DEVICE)
print("Repository Root:", repo_root)
"""))

# Cell 2: Section 1 - Data Ingestion, Ring Slicing & VU13P Hardware Limits
cells.append(nbf.v4.new_markdown_cell(r"""## 1. Ingestão de Dados Ringer (`v2` = 25 Anéis) e Limites de Recursos na FPGA VU13P

Configuramos a região cinemática $(\text{iet}=5, \text{ieta}=0)$, aplicamos o fatiamento físico da topologia `v2` (25 anéis internos do calorímetro) com normalização $L_1$, e definimos o teto de recursos da instância na FPGA **AMD/Xilinx Virtex UltraScale+ VU13P (Tabela 15)**: uma única rede, que divide o chip com outros algoritmos do trigger, recebe no máximo 1 % de cada recurso, dividido pela margem do modelo de custo calibrado ($B_j = 0{,}01\,B_{j,\text{chip}} / (1 + m_j)$).
"""))

cells.append(nbf.v4.new_code_cell(r"""# ==============================================================================
# PARÂMETROS GLOBAIS DO EXPERIMENTO
# ==============================================================================
EXPORT_DIR = os.path.join(repo_root, "results")
DATASET_PREFIX = "mc21_13p6TeV.Zee.JF17.2sigma.5M"
MODEL_NAME = "QCNN1D_KKT"
IET = 5
IETA = 0
VERSION = 2        # 0 (100 anéis), 1 (50 anéis), ou 2 (25 anéis)
N_SPLITS = 5       # Folds para validação cruzada
EPOCHS_CV = 25     # Épocas por fold com Warm-Start e ReduceLROnPlateau
QAT_EPOCHS = 8     # Épocas de QAT por candidato no laço de alocação

# Dados: diretório com os arquivos <DATASET_PREFIX>.et<IET>_eta<IETA>.h5 (variável RINGER_DATA_DIR)
DATA_DIR = os.environ.get("RINGER_DATA_DIR", os.path.join(repo_root, "data"))

# Precisões fixas neste laço: entrada sem sinal ufixed<8,0> e ReLUs de 8 bits
INPUT_BITS = 8
ACT_BITS = {"conv1d_1": 8, "dense_1": 8}

# Critério de seleção: menor custo cujo SP de validação fica no máximo 0,1 p.p. abaixo do 8 bits uniforme
SP_TOLERANCE = 1e-3
MAX_BUDGET_STEPS = 8      # passos da bisseção no orçamento de LUT
BUDGET_REL_TOL = 0.03     # para quando o intervalo de orçamento fica abaixo de 3 %
INSTANCE_FRACTION = 0.01  # teto: 1 % de cada recurso da VU13P

cfg_versao = CONFIG_VERSOES[VERSION]
input_dim = cfg_versao["input_dim"]
kernel_size = cfg_versao["kernel"]
num_filters = cfg_versao["filters"]
dense_units = cfg_versao["dense_units"]
ARCH = {"input_dim": input_dim, "filters": num_filters, "kernel": kernel_size, "dense_units": dense_units}

# Teto da instância: 1 % do chip por recurso, dividido pela margem do modelo de custo calibrado
COST_MODEL = get_cost_model()
VU13P_INSTANCE_BUDGET = get_vu13p_budget(
    {k: INSTANCE_FRACTION for k in VU13P_LIMITS}, margins=COST_MODEL.margins
)

df_vu13p = pd.DataFrame([
    {
        "Recurso FPGA (VU13P)": k,
        "Capacidade Total Chip (Tabela 15)": f"{VU13P_LIMITS[k]:,.0f}",
        "1 % do Chip": f"{INSTANCE_FRACTION * VU13P_LIMITS[k]:,.0f}",
        "Margem do Modelo de Custo": f"{COST_MODEL.margins.get(k, 0.0) * 100:.1f} %",
        "Orçamento da Instância (B_j)": f"{VU13P_INSTANCE_BUDGET[k]:,.0f}",
    }
    for k in VU13P_LIMITS.keys()
])
display(df_vu13p)

exporter = ArtifactExporter(
    export_dir=EXPORT_DIR,
    dataset_prefix=DATASET_PREFIX,
    model_name=MODEL_NAME,
    iet=IET,
    ieta=IETA,
    version=VERSION,
)

data_path = os.path.join(DATA_DIR, f"{DATASET_PREFIX}.et{IET}_eta{IETA}.h5")
df_loaded = pd.read_hdf(data_path, key="data")

ring_columns = sorted(
    [col for col in df_loaded.columns if col.startswith("trig_L2_cl_ring_")],
    key=lambda x: int(x.split("_")[-1]),
)
signal_df = df_loaded[df_loaded["target"] == 1][ring_columns]
background_df = df_loaded[df_loaded["target"] == 0][ring_columns]
ns = min(len(signal_df), len(background_df))

sig_sliced, estrutura_v = slice_rings_by_version(signal_df.values[:ns], VERSION)
bkg_sliced, _ = slice_rings_by_version(background_df.values[:ns], VERSION)
sig_normed = norm1(sig_sliced)
bkg_normed = norm1(bkg_sliced)

print(f"Total de eventos balanceados: {ns * 2:,} ({ns:,} Elétrons / {ns:,} Jatos)")
print(f"Estrutura das camadas em v{VERSION} ({input_dim} anéis): {estrutura_v}")

# Visualização 1: Perfil Médio de Energia por Camada
plot_rings_profile(
    sig_normed=sig_normed,
    bkg_normed=bkg_normed,
    estrutura=estrutura_v,
    version=VERSION,
    iet=IET,
    ieta=IETA,
    save_path=exporter.get_plot_path(f"rings_profile_v{VERSION}.png"),
    show=True,
)

X_all = np.vstack([sig_normed, bkg_normed]).astype(np.float32)
y_all = np.concatenate([np.ones(ns, dtype=np.float32), np.zeros(ns, dtype=np.float32)])

X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=SEED
)
print(f"Conjunto de Desenvolvimento (CV): {X_dev.shape} | Conjunto Holdout Isolado: {X_holdout.shape}")
"""))

# Cell 3: Section 2 - 3-Stage Closed-Loop Optimization (QKeras -> SP/VU13P -> Lagrange/KKT)
cells.append(nbf.v4.new_markdown_cell(r"""## 2. Ciclo Fechado de Otimização em 3 Etapas: QKeras $\to$ SP & VU13P $\to$ Lagrange/KKT

Objetivo: **o menor consumo de recursos que mantém o SP** do 8 bits uniforme (tolerância de 0,1 p.p. na validação), dentro do teto da instância. A latência não entra como restrição: no domínio calibrado (hls4ml `io_parallel`, `ReuseFactor=1`) ela fica entre 9 e 12 ciclos e não cresce quando os bits caem.

Para um orçamento de LUT $B$, o subproblema usa a sensibilidade de Hessiana $\Omega(\mathbf{w}) = \sum_l c_l 4^{-w_l}$ como substituto de $1 - SP$:

$$\min_{\mathbf{w}} \sum_l c_l 4^{-w_l} \quad \text{s.a.} \quad \tilde{g}_j(\mathbf{w}) = \frac{R_j(\mathbf{w}) - B_j}{B_j} \le 0, \qquad R_j \text{ do modelo de custo calibrado}$$

Em cada passo $t$:
1. **Lagrange/KKT**: para o orçamento $B^{(t)}$, `compute_primal_bit_update` acha os preços-sombra $\mu_j$ por ascensão dual até as condições KKT e devolve o ótimo inteiro;
2. **QKeras**: o candidato é treinado com QAT (*warm-start* do FP32, mesma semente para todos);
3. **SP e custo**: mede o SP de validação e recalcula os recursos com os pesos treinados. Se o SP se mantém e cabe no teto, o orçamento desce; se não, sobe (bisseção em $\log B$).
"""))

cells.append(nbf.v4.new_code_cell(r"""# Split de calibração para o laço de alocação
X_tr_ref, X_val_ref, y_tr_ref, y_val_ref = train_test_split(
    X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=SEED
)

# Modelo de referência FP32: warm-start do QAT, sensibilidade de 2ª ordem e custo dos candidatos
model_fp32, sp_cb_fp32, history_fp32 = train_fp32(
    ARCH, X_tr_ref, y_tr_ref, X_val_ref, y_val_ref, epochs=EPOCHS_CV, seed=SEED
)
hw_fp32 = compute_model_hardware_costs(model_fp32, input_dim=input_dim, input_bits=INPUT_BITS)

fp32_ref_eval = evaluate_model(model_fp32, X_val_ref, y_val_ref, X_holdout, y_holdout)
print(
    f"\n[Referência FP32] SP_val = {fp32_ref_eval['sp_val']*100:.2f}% | holdout no limiar da validação: "
    f"SP = {fp32_ref_eval['sp_test']*100:.2f}% | PD = {fp32_ref_eval['pd_test']*100:.2f}% | FA = {fp32_ref_eval['fa_test']*100:.2f}%"
)

# Sensibilidade de Hutchinson Tr(H_l) -> c_l (traço total por camada, sorteios por camada)
rng_sens = np.random.default_rng(SEED)
sample_idx = rng_sens.choice(len(X_val_ref), size=min(4096, len(X_val_ref)), replace=False)
x_sample_tf = tf.convert_to_tensor(X_val_ref[sample_idx])
y_sample_tf = tf.convert_to_tensor(y_val_ref[sample_idx], dtype=tf.float32)[:, None]
bce_loss = keras.losses.BinaryCrossentropy()

traces = compute_hutchinson_hessian_trace(model_fp32, bce_loss, x_sample_tf, y_sample_tf, num_hvp=32, seed=SEED)
c_coeffs = compute_layer_c_coefficients(model_fp32, traces)

layer_names = [item["layer_name"] for item in hw_fp32["breakdown"]]
macs_list = [float(item["macs"]) for item in hw_fp32["breakdown"]]
params_list = [float(item["params"]) for item in hw_fp32["breakdown"]]
c_list = [c_coeffs[name] for name in layer_names]
bops_uniform_8bit = sum(m * 64.0 for m in macs_list)
print("Tr(H_l):", {k: f"{v:.4g}" for k, v in traces.items()})
print("c_l    :", {k: f"{v:.4g}" for k, v in c_coeffs.items()})

# Custo de cada candidato antes do treino: pesos FP32 quantizados no modelo calibrado
cost_fn = make_cost_fn(model_fp32, input_bits=INPUT_BITS)


def train_qat_candidate(w_bits):
    # Etapa QKeras: mesma semente e mesmo warm-start para todos os candidatos (comparação pareada)
    q_cand, cb_cand, _ = train_qat(
        ARCH, w_bits, ACT_BITS, INPUT_BITS, model_fp32,
        X_tr_ref, y_tr_ref, X_val_ref, y_val_ref,
        epochs=QAT_EPOCHS, learning_rate=8e-4, patience=5, seed=SEED, reduce_lr=False,
    )
    return float(cb_cand.best_sp), q_cand


evaluated = {}


def evaluate_candidate(w_bits):
    # Etapa SP/VU13P: SP de validação e recursos calibrados com os pesos treinados
    key = tuple(int(w_bits[name]) for name in layer_names)
    if key not in evaluated:
        sp_val, q_cand = train_qat_candidate(w_bits)
        hw_cand = compute_model_hardware_costs(
            q_cand, weights_bits=w_bits, act_bits=ACT_BITS, input_dim=input_dim, input_bits=INPUT_BITS
        )
        evaluated[key] = {
            "w_bits": {name: int(w_bits[name]) for name in layer_names},
            "sp_val": sp_val,
            "resources": {k: float(v) for k, v in hw_cand["vu13p_resources"].items()},
        }
    return evaluated[key]


def fits_instance(resources):
    return all(resources[r] <= VU13P_INSTANCE_BUDGET[r] for r in ("DSP", "LUT", "FF"))


# Referência de precisão: 8 bits uniforme treinado da mesma forma
w_8bit = {name: 8 for name in layer_names}
ref_8bit = evaluate_candidate(w_8bit)
SP_TARGET = ref_8bit["sp_val"] - SP_TOLERANCE
print(
    f"\n[8 bits uniforme] SP_val = {ref_8bit['sp_val']*100:.2f}% | LUT = {ref_8bit['resources']['LUT']:,.0f} | "
    f"FF = {ref_8bit['resources']['FF']:,.0f} | DSP = {ref_8bit['resources']['DSP']:.0f} -> alvo SP_val >= {SP_TARGET*100:.2f}%"
)

# Etapa Lagrange/KKT dentro de uma bisseção no orçamento de LUT
kkt_optimizer = LagrangeKKTQuantizerOptimizer(hardware_limits=dict(VU13P_INSTANCE_BUDGET), lr_mu=0.80)
lut_low = cost_fn({name: 2 for name in layer_names}, ACT_BITS)["LUT"]
lut_high = min(VU13P_INSTANCE_BUDGET["LUT"], cost_fn(w_8bit, ACT_BITS)["LUT"])
best = None
accepted = []
sp_history_kkt = []

print(f"\n=== Bisseção no orçamento de LUT: [{lut_low:,.0f}, {lut_high:,.0f}] ===")
for t_step in range(1, MAX_BUDGET_STEPS + 1):
    budget_lut = float(np.sqrt(lut_low * lut_high))
    kkt_optimizer.limits["LUT"] = budget_lut
    w_cfg, a_cfg, kkt_sol = kkt_optimizer.compute_primal_bit_update(
        layer_names=layer_names,
        c_coeffs=c_list,
        macs_list=macs_list,
        params_list=params_list,
        current_w=w_8bit,
        current_a=ACT_BITS,
        b_min=2,
        b_max=8,
        input_bits=INPUT_BITS,
        cost_fn=cost_fn,
    )
    cand = evaluate_candidate(w_cfg)
    sp_t, res_t = cand["sp_val"], cand["resources"]
    sp_history_kkt.append(sp_t)

    lag_cost_t, g_val_t = kkt_optimizer.lagrangian_penalty(sp_loss=1.0 - sp_t, resource_usage=res_t)
    _, kkt_diag_t = check_kkt_stopping_criteria(
        sp_history=sp_history_kkt,
        g_val=g_val_t,
        mu=kkt_optimizer.mu,
        tol_sp=1e-3,
        tol_kkt=2e-2,
        window=2,
        stationarity_residual=kkt_sol["stationarity_residual"],
    )
    kkt_optimizer.record_step(
        iteration=t_step,
        sp_val=sp_t,
        w_bits=w_cfg,
        a_bits=a_cfg,
        resource_usage=res_t,
        g_val=g_val_t,
        lagrangian_cost=lag_cost_t,
        kkt_diag=kkt_diag_t,
    )

    keeps_sp = sp_t >= SP_TARGET
    accepted_t = keeps_sp and fits_instance(res_t)
    if accepted_t:
        accepted.append(cand)
        lut_high = budget_lut
        if best is None or res_t["LUT"] < best["resources"]["LUT"]:
            best = {**cand, "kkt_solution": kkt_sol, "budget_lut": budget_lut, "iteration": t_step}
    else:
        lut_low = budget_lut

    print(
        f"  [Passo {t_step}] B_LUT={budget_lut:,.0f} | w={list(w_cfg.values())}b | "
        f"μ_LUT={kkt_optimizer.mu['LUT']:.3g} | KKT resíduo={kkt_sol['stationarity_residual']:.1e} | "
        f"SP={sp_t*100:.2f}% | LUT={res_t['LUT']:,.0f}, FF={res_t['FF']:,.0f}, DSP={res_t['DSP']:.0f} | "
        f"{'aceito' if accepted_t else 'rejeitado'}"
    )
    if lut_high / lut_low < 1.0 + BUDGET_REL_TOL:
        break

if best is None:
    feasible = [c for c in evaluated.values() if fits_instance(c["resources"])]
    best = max(feasible, key=lambda c: c["sp_val"])
    best = {**best, "kkt_solution": {}, "budget_lut": None, "iteration": None}
    print("Nenhum candidato manteve o SP; usando o de maior SP dentro do teto.")

kkt_weights_bits = dict(best["w_bits"])
kkt_act_bits = dict(ACT_BITS)
print(
    f"\nEscolhido: w={list(kkt_weights_bits.values())}b, a={list(kkt_act_bits.values())}b, entrada {INPUT_BITS}b | "
    f"SP_val = {best['sp_val']*100:.2f}% | LUT = {best['resources']['LUT']:,.0f} "
    f"({best['resources']['LUT'] / ref_8bit['resources']['LUT'] * 100:.0f}% do 8 bits uniforme)"
)

kkt_result = {
    **best["kkt_solution"],
    "integer_bits": kkt_weights_bits,
    "activation_bits": kkt_act_bits,
    "input_bits": INPUT_BITS,
    "vu13p_instance_budget": VU13P_INSTANCE_BUDGET,
    "selection": {
        "sp_tolerance": SP_TOLERANCE,
        "sp_target": SP_TARGET,
        "reference_8bit": ref_8bit,
        "chosen_budget_lut": best["budget_lut"],
        "chosen_iteration": best["iteration"],
        "candidates": list(evaluated.values()),
    },
    "primal_dual_history": kkt_optimizer.history,
    "final_multipliers_mu": best["kkt_solution"].get("shadow_prices", kkt_optimizer.mu),
    "kkt_stopping_diagnostics": kkt_optimizer.history[-1]["kkt_diagnostics"],
}

# Visualização 2: Trajetória Primal-Dual KKT e Preços de Sombra (μ_j)
plot_primal_dual_kkt_evolution(
    kkt_history=kkt_optimizer.history,
    save_path=exporter.get_plot_path("kkt_primal_dual_evolution.png"),
    title_prefix=f"VU13P [v{VERSION}]",
    show=True,
)
"""))

# Cell 4: Section 3 - Paired 5-fold cross-validation and holdout evaluation
cells.append(nbf.v4.new_markdown_cell(r"""## 3. Confirmação em Validação Cruzada Pareada ($K=5$ Folds) e Avaliação Holdout

Protocolo:
* em cada fold, o FP32 que inicializa o QAT é treinado **só com o treino daquele fold** (as validações dos folds não podem ter sido vistas pelo *warm-start*);
* FP32, 8 bits uniforme e a alocação escolhida são treinados nos **mesmos folds, com a mesma semente e o mesmo warm-start**, e comparados pela diferença pareada de SP por fold (média, desvio e erro padrão);
* no holdout, cada modelo é avaliado no **limiar escolhido na sua validação** e o resultado é a média dos folds (escolher o joelho no próprio holdout ou relatar só o melhor fold é otimista).

A alocação é confirmada se a diferença pareada média em relação ao 8 bits uniforme for $\ge -0{,}1$ p.p.; se não for, testa-se o próximo candidato aceito no laço da Seção 2, em ordem de custo.
"""))

cells.append(nbf.v4.new_code_cell(r"""skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(X_dev, y_dev))

# Referências por fold: FP32 (só treino do fold) e 8 bits uniforme com warm-start desse FP32
fold_refs = {}
print(f"Treinando referências por fold (FP32 e 8 bits uniforme, {N_SPLITS} folds)...")
for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits, start=1):
    X_tr_f, y_tr_f, X_val_f, y_val_f = X_dev[tr_idx], y_dev[tr_idx], X_dev[val_idx], y_dev[val_idx]
    fp32_f, _, _ = train_fp32(ARCH, X_tr_f, y_tr_f, X_val_f, y_val_f, epochs=EPOCHS_CV, seed=SEED + fold_idx)
    u8_f, cb_u8_f, hist_u8_f = train_qat(
        ARCH, w_8bit, ACT_BITS, INPUT_BITS, fp32_f, X_tr_f, y_tr_f, X_val_f, y_val_f,
        epochs=EPOCHS_CV, seed=SEED + fold_idx,
    )
    fold_refs[fold_idx] = {
        "fp32": {"model": fp32_f, "eval": evaluate_model(fp32_f, X_val_f, y_val_f, X_holdout, y_holdout)},
        "u8": {"model": u8_f, "eval": evaluate_model(u8_f, X_val_f, y_val_f, X_holdout, y_holdout)},
    }
    print(
        f"  [Fold {fold_idx}] SP_val FP32 = {fold_refs[fold_idx]['fp32']['eval']['sp_val']*100:.2f}% | "
        f"8 bits = {fold_refs[fold_idx]['u8']['eval']['sp_val']*100:.2f}%"
    )


def cross_validate(w_bits):
    runs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits, start=1):
        X_tr_f, y_tr_f, X_val_f, y_val_f = X_dev[tr_idx], y_dev[tr_idx], X_dev[val_idx], y_dev[val_idx]
        q_f, cb_f, hist_f = train_qat(
            ARCH, w_bits, ACT_BITS, INPUT_BITS, fold_refs[fold_idx]["fp32"]["model"],
            X_tr_f, y_tr_f, X_val_f, y_val_f, epochs=EPOCHS_CV, seed=SEED + fold_idx,
        )
        runs.append({
            "fold": fold_idx, "model": q_f, "best_epoch": cb_f.best_epoch + 1, "history": hist_f,
            "y_val": y_val_f, "eval": evaluate_model(q_f, X_val_f, y_val_f, X_holdout, y_holdout),
        })
    return runs


def sp_list(source, key="sp_val"):
    return [run["eval"][key] for run in source]


u8_runs = [{"fold": k, "eval": fold_refs[k]["u8"]["eval"]} for k in fold_refs]
fp32_runs = [{"fold": k, "eval": fold_refs[k]["fp32"]["eval"]} for k in fold_refs]

# Candidatos aceitos no laço, do mais barato ao mais caro
cv_candidates = []
for cand in sorted(accepted, key=lambda c: c["resources"]["LUT"]):
    if cand["w_bits"] not in [c["w_bits"] for c in cv_candidates]:
        cv_candidates.append(cand)

kkt_runs, cv_decision = None, None
for cand in cv_candidates:
    runs = cross_validate(cand["w_bits"])
    diff_u8 = paired_difference(sp_list(runs), sp_list(u8_runs))
    passed = diff_u8["mean"] >= -SP_TOLERANCE
    print(
        f"[CV] w={list(cand['w_bits'].values())}b: SP_val = {np.mean(sp_list(runs))*100:.2f}% | "
        f"Δ vs 8 bits (pareado) = {diff_u8['mean']*100:+.2f} ± {diff_u8['std']*100:.2f} p.p. "
        f"(erro padrão {diff_u8['se']*100:.2f}) -> {'confirmado' if passed else 'reprovado'}"
    )
    if passed:
        kkt_runs, chosen_cand = runs, cand
        cv_decision = {"w_bits": cand["w_bits"], "paired_diff_vs_u8": diff_u8}
        break

if kkt_runs is None:
    print("Nenhuma alocação mista manteve o SP em CV pareada: fica o 8 bits uniforme.")
    kkt_runs = [{"fold": k, "model": fold_refs[k]["u8"]["model"], "best_epoch": None, "history": {},
                 "y_val": y_dev[fold_splits[k - 1][1]], "eval": fold_refs[k]["u8"]["eval"]} for k in fold_refs]
    chosen_cand = evaluated[tuple(w_8bit[n] for n in layer_names)]
    cv_decision = {"w_bits": w_8bit, "paired_diff_vs_u8": paired_difference(sp_list(kkt_runs), sp_list(u8_runs))}

kkt_weights_bits = dict(chosen_cand["w_bits"])
diff_kkt_u8 = paired_difference(sp_list(kkt_runs), sp_list(u8_runs))
diff_kkt_fp32 = paired_difference(sp_list(kkt_runs), sp_list(fp32_runs))
diff_ho_kkt_u8 = paired_difference(sp_list(kkt_runs, "sp_test"), sp_list(u8_runs, "sp_test"))

paired_df = pd.DataFrame([
    {
        "fold": k,
        "sp_val_fp32": fold_refs[k]["fp32"]["eval"]["sp_val"],
        "sp_val_u8": fold_refs[k]["u8"]["eval"]["sp_val"],
        "sp_val_kkt": run["eval"]["sp_val"],
        "diff_val_kkt_u8": run["eval"]["sp_val"] - fold_refs[k]["u8"]["eval"]["sp_val"],
        "diff_val_kkt_fp32": run["eval"]["sp_val"] - fold_refs[k]["fp32"]["eval"]["sp_val"],
        "sp_holdout_fp32": fold_refs[k]["fp32"]["eval"]["sp_test"],
        "sp_holdout_u8": fold_refs[k]["u8"]["eval"]["sp_test"],
        "sp_holdout_kkt": run["eval"]["sp_test"],
    }
    for k, run in zip(fold_refs, kkt_runs)
])
paired_df.to_csv(os.path.join(exporter.metadata_dir, "paired_cv_comparison.csv"), index=False)
display((paired_df.set_index("fold") * 100).round(3))

print(f"\nAlocação confirmada: w={list(kkt_weights_bits.values())}b, a={list(ACT_BITS.values())}b, entrada {INPUT_BITS}b")
print(f"  SP_val (CV)          : FP32 {np.mean(sp_list(fp32_runs))*100:.2f}% | 8 bits {np.mean(sp_list(u8_runs))*100:.2f}% | KKT {np.mean(sp_list(kkt_runs))*100:.2f}% ± {np.std(sp_list(kkt_runs), ddof=1)*100:.2f}%")
print(f"  Δ pareado (validação): KKT − 8 bits = {diff_kkt_u8['mean']*100:+.2f} ± {diff_kkt_u8['std']*100:.2f} p.p. (erro padrão {diff_kkt_u8['se']*100:.2f}) | KKT − FP32 = {diff_kkt_fp32['mean']*100:+.2f} ± {diff_kkt_fp32['std']*100:.2f} p.p.")
print(f"  Holdout (limiar da validação, média dos folds): FP32 {np.mean(sp_list(fp32_runs, 'sp_test'))*100:.2f}% | 8 bits {np.mean(sp_list(u8_runs, 'sp_test'))*100:.2f}% | KKT {np.mean(sp_list(kkt_runs, 'sp_test'))*100:.2f}% (Δ KKT − 8 bits = {diff_ho_kkt_u8['mean']*100:+.2f} p.p.)")

# Artefatos por fold da alocação confirmada
training_results_metadata, cv_records, folds_y_true, folds_y_pred = [], [], [], []
for run in kkt_runs:
    ev = run["eval"]
    fold_model_path = exporter.save_fold_model(run["model"], fold=run["fold"], repeat=1)
    folds_y_true.append(run["y_val"])
    folds_y_pred.append(ev["y_pred_val"])
    record = {
        "fold": run["fold"],
        "repeat": 1,
        "sp_max": ev["sp_val"],
        "pd_knee": ev["pd_val"],
        "fa_knee": ev["fa_val"],
        "threshold_knee": ev["threshold"],
        "auc": ev["auc_val"],
        "best_epoch": run["best_epoch"],
        "model_path": os.path.basename(fold_model_path),
        "sp_holdout_val_threshold": ev["sp_test"],
        "pd_holdout_val_threshold": ev["pd_test"],
        "fa_holdout_val_threshold": ev["fa_test"],
    }
    cv_records.append(record)
    training_results_metadata.append({**{k: v for k, v in record.items() if k != "model_path"},
                                      "max_sp": ev["sp_val"], "keras_history": run["history"]})
cv_df = pd.DataFrame(cv_records)

# Modelo exportado: o fold de maior SP de validação; as métricas relatadas são as médias acima
best_run = max(kkt_runs, key=lambda run: run["eval"]["sp_val"])
best_fold_idx = best_run["fold"]
best_qmodel = best_run["model"]
best_history_dict = best_run["history"]
best_eval = best_run["eval"]
fp32_best_eval = fold_refs[best_fold_idx]["fp32"]["eval"]
y_pred_holdout, thresh_ho = best_eval["y_pred_test"], best_eval["threshold"]
sp_ho, pd_ho, fa_ho = best_eval["sp_test"], best_eval["pd_test"], best_eval["fa_test"]
y_pred_fp32_holdout = fp32_best_eval["y_pred_test"]
sp_fp32_ho, pd_fp32_ho, fa_fp32_ho = fp32_best_eval["sp_test"], fp32_best_eval["pd_test"], fp32_best_eval["fa_test"]

hw_qkeras_kkt = compute_model_hardware_costs(
    best_qmodel, weights_bits=kkt_weights_bits, act_bits=ACT_BITS, input_dim=input_dim, input_bits=INPUT_BITS
)
hw_u8 = compute_model_hardware_costs(
    fold_refs[best_fold_idx]["u8"]["model"], weights_bits=w_8bit, act_bits=ACT_BITS,
    input_dim=input_dim, input_bits=INPUT_BITS,
)
print(f"\n[Modelo exportado: fold #{best_fold_idx}] holdout no limiar da validação: SP = {sp_ho*100:.2f}% | PD = {pd_ho*100:.2f}% | FA = {fa_ho*100:.2f}% | limiar = {thresh_ho:.4f}")
print(f"  • Recursos VU13P: {', '.join(f'{k} = {v:,.0f}' for k, v in hw_qkeras_kkt['vu13p_resources'].items())}")

kkt_result["integer_bits"] = kkt_weights_bits
kkt_result["cv_confirmation"] = {
    **cv_decision,
    "paired_diff_vs_fp32": diff_kkt_fp32,
    "paired_diff_holdout_vs_u8": diff_ho_kkt_u8,
    "candidates_tested": [c["w_bits"] for c in cv_candidates[: cv_candidates.index(chosen_cand) + 1]] if chosen_cand in cv_candidates else [],
}

def mean_eval(runs, key):
    return float(np.mean(sp_list(runs, key)))

# Exportação Hierárquica Jubiaba
exporter.save_best_model(best_qmodel, input_dim=input_dim)
exporter.save_history(best_history_dict)
exporter.save_evaluation_splits(X_holdout, y_holdout, y_pred_holdout, threshold=thresh_ho)
exporter.save_metadata(
    training_results=training_results_metadata,
    cv_df=cv_df,
    kkt_result=kkt_result,
    extra_config={
        "rings_structure": estrutura_v,
        "vu13p_chip_limits": VU13P_LIMITS,
        "vu13p_instance_budget": VU13P_INSTANCE_BUDGET,
        "hardware_fp32": hw_fp32,
        "hardware_uniform_8bit": hw_u8,
        "hardware_qkeras_kkt": hw_qkeras_kkt,
        "holdout_metrics": {
            "aggregation": "mean over the CV fold models, each at the threshold chosen on its validation fold",
            "sp": mean_eval(kkt_runs, "sp_test"),
            "pd": mean_eval(kkt_runs, "pd_test"),
            "fa": mean_eval(kkt_runs, "fa_test"),
            "threshold": thresh_ho,
            "fp32_sp": mean_eval(fp32_runs, "sp_test"),
            "fp32_pd": mean_eval(fp32_runs, "pd_test"),
            "fp32_fa": mean_eval(fp32_runs, "fa_test"),
            "u8_sp": mean_eval(u8_runs, "sp_test"),
            "u8_pd": mean_eval(u8_runs, "pd_test"),
            "u8_fa": mean_eval(u8_runs, "fa_test"),
        },
        "paired_cv": paired_df.to_dict(orient="records"),
    },
    seed=SEED,
    folds_y_true=folds_y_true,
    folds_y_pred=folds_y_pred,
)
"""))

# Cell 5: Section 4 - Diagnostic & Physics Visualizations
cells.append(nbf.v4.new_markdown_cell(r"""## 4. Diagnóstico Visual de Recursos na VU13P e Desempenho Físico (`codeTCC` + `Jubiaba`)

Geramos os gráficos de utilização de recursos na FPGA Virtex UltraScale+ VU13P (8 bits uniforme e alocação escolhida, ambos com pesos treinados), curvas de aprendizado, curvas ROC (modelo exportado, duelo FP32 vs. KKT do mesmo fold e envoltória dos 5 folds), distribuição de scores, matriz de confusão e boxplots de estabilidade. Os gráficos de holdout usam o limiar escolhido na validação.
"""))

cells.append(nbf.v4.new_code_cell(r"""# 3. Ocupação de Recursos na FPGA Virtex UltraScale+ VU13P e Alocação de Bits por Camada
plot_vu13p_resource_utilization(
    hw_8bit=hw_u8,
    hw_kkt=hw_qkeras_kkt,
    vu13p_budget=VU13P_INSTANCE_BUDGET,
    save_path=exporter.get_plot_path("vu13p_resource_utilization.png"),
    title_prefix=f"VU13P [v{VERSION}]",
    show=True,
)

# 4. Curvas de Aprendizado da Melhor Execução
plot_learning_curves(
    history_dict=best_history_dict,
    save_path=exporter.get_plot_path("learning_curves.png"),
    title_prefix=f"[v{VERSION}]",
    show=True,
)

# 5. Curva ROC do Melhor Modelo (Global + Zoom Operacional)
plot_best_model_roc(
    y_true=y_holdout,
    y_pred=y_pred_holdout,
    sp_val=sp_ho,
    pd_knee=pd_ho,
    fa_knee=fa_ho,
    save_path=exporter.get_plot_path("roc_best_model.png"),
    title=f"Curva ROC Holdout (fold #{best_fold_idx}, limiar da validação) - QCNN1D KKT v{VERSION}",
    show=True,
)

# 6. Duelo ROC: Referência FP32 vs. QKeras Lagrange/KKT (Global + Zoom nos Working Points ATLAS)
plot_roc_duel_fp32_vs_kkt(
    y_val=y_holdout,
    y_pred_fp32=y_pred_fp32_holdout,
    sp_fp32=sp_fp32_ho,
    pd_fp32=pd_fp32_ho,
    fa_fp32=fa_fp32_ho,
    y_pred_q=y_pred_holdout,
    sp_q=sp_ho,
    pd_q=pd_ho,
    fa_q=fa_ho,
    bops_ratio=hw_qkeras_kkt["total_bops"] / bops_uniform_8bit,
    save_path=exporter.get_plot_path("roc_duel_fp32_vs_kkt.png"),
    title_prefix=f"QCNN1D KKT v{VERSION}",
    show=True,
)

# 7. Envoltória ROC da Validação Cruzada
plot_cv_roc_envelope(
    folds_y_true=folds_y_true,
    folds_y_pred=folds_y_pred,
    save_path=exporter.get_plot_path("roc_cv_envelope.png"),
    title=f"Envoltória ROC ({N_SPLITS}-Fold CV) - QCNN1D KKT v{VERSION}",
    show=True,
)

# 8. Histograma de Scores da Saída (Linear e Log)
plot_output_score_histogram(
    y_true=y_holdout,
    y_pred=y_pred_holdout,
    threshold_knee=thresh_ho,
    save_path=exporter.get_plot_path("output_scores_hist.png"),
    title=f"Scores de Saída Holdout - v{VERSION}",
    show=True,
)

# 9. Matriz de Confusão no Ponto de Joelho (SP Knee)
plot_confusion_matrix_knee(
    y_true=y_holdout,
    y_pred=y_pred_holdout,
    threshold_knee=thresh_ho,
    save_path=exporter.get_plot_path("confusion_matrix_knee.png"),
    title=f"Matriz de Confusão Holdout - QCNN1D KKT v{VERSION}",
    show=True,
)

# 10. Boxplots de Estabilidade em Validação Cruzada e Falso Alarme por Workpoint
plot_cv_statistics_boxplots(
    cv_df=cv_df,
    save_path=exporter.get_plot_path("cv_boxplots.png"),
    title=f"Estabilidade em Validação Cruzada ({N_SPLITS} Folds) - Topologia v{VERSION} ({input_dim} Anéis)",
    show=True,
)

plot_fa_per_fold_workpoints(
    folds_y_true=folds_y_true,
    folds_y_pred=folds_y_pred,
    save_path=exporter.get_plot_path("fa_per_fold_workpoints.png"),
    title=f"Falso Alarme por Workpoint do Trigger ATLAS ({N_SPLITS} Folds) - v{VERSION}",
    show=True,
)
"""))

# Cell 6: Section 5 - Final Quantization Report & Jubiaba Tree Inspection
cells.append(nbf.v4.new_markdown_cell(r"""## 5. Tabela de Quantização Final por Camada, Recursos na VU13P e Árvore Jubiaba

Consolidamos a tabela detalhada de precisão $(w_l^*, a_l^*, n_l)$ por camada `QConv1D`/`QDense`, o comparativo global de recursos na FPGA **Virtex UltraScale+ VU13P** e a listagem dos artefatos exportados.
"""))

cells.append(nbf.v4.new_code_cell(r"""layer_quant_rows = []
for item in hw_qkeras_kkt["breakdown"]:
    layer_quant_rows.append({
        "Camada QKeras": item["layer_name"],
        "Tipo": item["layer_type"],
        "Filtros/Unidades (n_l)": item["units"],
        "Bits Pesos (w_l)": item["b_w"],
        "Bits da Entrada (a_in)": item["b_a"],
        "Parâmetros": item["params"],
        "MACs": item["macs"],
        "BOPs": item["bops"],
        "VU13P DSPs": int(item["DSP"]),
        "VU13P LUTs": f"{item['LUT']:.1f}",
        "VU13P FFs": f"{item['FF']:.1f}",
        "VU13P BRAM [bits]": int(item["BRAM_bits"]),
    })

df_layer_quant = pd.DataFrame(layer_quant_rows)
print("=== Tabela de Quantização Final por Camada e Consumo na FPGA VU13P ===")
display(df_layer_quant)

res_u8 = hw_u8["vu13p_resources"]
res_kkt = hw_qkeras_kkt["vu13p_resources"]


def summary_row(name, precision, runs, diff, hw):
    sp_val = sp_list(runs)
    sp_ho_list = sp_list(runs, "sp_test")
    res = hw["vu13p_resources"] if hw else None
    return {
        "Modelo": name,
        "Precisão (w_l / a_l)": precision,
        "SP CV val [%]": f"{np.mean(sp_val)*100:.2f} ± {np.std(sp_val, ddof=1)*100:.2f}",
        "SP Holdout [%]": f"{np.mean(sp_ho_list)*100:.2f} ± {np.std(sp_ho_list, ddof=1)*100:.2f}",
        "Δ SP vs 8 bits [p.p.]": "—" if diff is None else f"{diff['mean']*100:+.2f} ± {diff['se']*100:.2f}",
        "BOPs": "—" if hw is None else f"{hw['total_bops']:,}",
        "VU13P DSPs": "—" if res is None else int(res["DSP"]),
        "VU13P LUTs": "—" if res is None else f"{res['LUT']:,.0f}",
        "VU13P FFs": "—" if res is None else f"{res['FF']:,.0f}",
    }


summary_df = pd.DataFrame([
    summary_row("FP32 (referência)", "ponto flutuante", fp32_runs,
                paired_difference(sp_list(fp32_runs), sp_list(u8_runs)), None),
    summary_row("8 bits uniforme", "w: [8, 8, 8]b | a: [8, 8]b", u8_runs, None, hw_u8),
    summary_row(
        "Alocação Lagrange/KKT",
        f"w: {list(kkt_weights_bits.values())}b | a: {list(ACT_BITS.values())}b",
        kkt_runs, diff_kkt_u8, hw_qkeras_kkt,
    ),
])
print("SP: média ± desvio dos 5 folds; holdout no limiar escolhido na validação; Δ pareado ± erro padrão.")
print("\n=== Comparativo Global de Desempenho Físico e Recursos na VU13P ===")
display(summary_df)

print(f"\n=== Árvore de Artefatos Gerada em: {exporter.base_dir} ===")
for root, dirs, files in os.walk(exporter.base_dir):
    level = root.replace(exporter.base_dir, "").count(os.sep)
    indent = "  " * level
    folder_name = os.path.basename(root) if level > 0 else f"v{VERSION}/"
    print(f"{indent}├── {folder_name}/")
    subindent = "  " * (level + 1)
    for f_name in sorted(files):
        f_path = os.path.join(root, f_name)
        size_kb = os.path.getsize(f_path) / 1024.0
        print(f"{subindent}└── {f_name} ({size_kb:.1f} KB)")
"""))

nb["cells"] = cells
notebook_path = "notebooks/01_quantized_cnn1d_optimization.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook gerado com sucesso em: {notebook_path}")
