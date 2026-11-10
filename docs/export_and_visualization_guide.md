# Guia de Estruturação Hierárquica (Padrão Jubiaba), Topologia `v2` (25 Anéis) e Galeria de Visualizações (`codeTCC` + `VU13P`)

## 1. Arquitetura de Exportação Hierárquica (Padrão Jubiaba)

Todos os artefatos produzidos pelo pipeline de otimização Primal-Dual Lagrange/KKT e treinamento ciente de quantização ([`src/exporter.py`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/exporter.py)) seguem rigorosamente a árvore de diretórios do projeto **Jubiaba**:

```text
<export_dir>/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/
├── saved_models/  # Todos os modelos por fold/repeat (model_v{V}_f{F}_r{R}.keras), best_model.keras, best_model.onnx
├── plots/         # Perfis de anéis, evolução KKT (μ_j), uso da VU13P, curvas de aprendizado, ROC, matriz de confusão e boxplots CV
├── history/       # Histórico por época da melhor execução em formatos CSV (training_history.csv) e JSON (history.json)
├── metadata/      # training_metadata_v{V}.pkl/.json, cross_validation_results.csv, reproducibility_metadata.json, kkt_allocation_results.json
└── evaluation/    # Splits de holdout comprimidos (holdout_test_set.npz) e predições (holdout_predictions.npz)
```

---

## 2. Fatiamento Facilitado de Anéis (`v0`, `v1`, `v2`)

A função [`slice_rings_by_version`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/metrics.py#L29-L60) e o dicionário [`CONFIG_VERSOES`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/metrics.py#L22-L26) permitem alternar com uma única variável (`VERSION = 0, 1, 2`) entre as três resoluções espaciais do calorímetro do ATLAS:

| Versão | Total de Anéis | Fator de Redução | Distribuição por Camada (`PS, EM1, EM2, EM3, HAD1, HAD2, HAD3`) | MACs Totais (CNN1D Compacta) |
| :--- | :---: | :---: | :---: | :---: |
| **`v0`** | **100** | $1\times$ | `8, 64, 8, 8, 4, 4, 4` | $7.608$ |
| **`v1`** | **50** | $2\times$ | `4, 32, 4, 4, 2, 2, 2` | $2.808$ |
| **`v2`** | **25** | $4\times$ | `2, 16, 2, 2, 1, 1, 1` | **$1.376$** |

---

## 3. Catálogo das 10 Visualizações Científicas (`src/visualization.py`)

O módulo [`src/visualization.py`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py) implementa todas as visualizações clássicas de `codeTCC` e da metodologia Primal-Dual Lagrange/KKT na FPGA Virtex UltraScale+ VU13P:

1. [`plot_rings_profile`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L26-L85) (`rings_profile_v2.png`): Perfil médio e desvio padrão da energia normalizada para Elétrons vs. Jatos com linhas divisórias das 7 camadas do calorímetro;
2. [`plot_primal_dual_kkt_evolution`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L569-L654) (`kkt_primal_dual_evolution.png`): Trajetória do ciclo fechado de 3 etapas mostrando a evolução do Índice-$SP$, dos Preços de Sombra ($\mu_{\text{DSP}}, \mu_{\text{LUT}}, \mu_{\text{FF}}, \mu_{\text{BRAM}}$) e das restrições normalizadas $g_j(\mathbf{x}) \le 0$ até o ponto ótimo KKT;
3. [`plot_vu13p_resource_utilization`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L499-L566) (`vu13p_resource_utilization.png`): Ocupação percentual dos limites $B_j$ da VU13P (`LUT`, `FF`, `BRAM`) e consumo detalhado de LUTs por camada $(w_l, a_l)$;
4. [`plot_learning_curves`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L88-L144) (`learning_curves.png`): Curvas de aprendizado de Loss (BCE), Acurácia e evolução das métricas do trigger ($SP, PD, 1-FA$);
5. [`plot_best_model_roc`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L147-L211) (`roc_best_model.png`): Curva ROC global e curva ROC com zoom na região de operação do trigger ($PD \in [94\%, 100\%]$) destacando os *Working Points* `Tight (96%)`, `Medium (97.5%)` e `Loose (98.5%)`;
6. [`plot_roc_duel_fp32_vs_kkt`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L434-L496) (`roc_duel_fp32_vs_kkt.png`): Comparativo direto da curva ROC global e zoom operacional entre a referência FP32 e o modelo QKeras otimizado por Lagrange/KKT;
7. [`plot_cv_roc_envelope`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L214-L259) (`roc_cv_envelope.png`): Curva ROC média e banda de incerteza $\pm 1\sigma$ sobre os 5 folds de validação cruzada;
8. [`plot_output_score_histogram`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L262-L300) (`output_scores_hist.png`): Histograma de densidade das saídas da rede (Sinal vs. Fundo) em escalas linear e logarítmica com marcação do limiar ótimo do joelho $SP$;
9. [`plot_confusion_matrix_knee`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L303-L347) (`confusion_matrix_knee.png`): Matriz de confusão percentual e absoluta avaliada exatamente no limiar do joelho $SP$;
10. [`plot_cv_statistics_boxplots`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L350-L384) (`cv_boxplots.png`) & [`plot_fa_per_fold_workpoints`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L387-L431) (`fa_per_fold_workpoints.png`): Boxplots de estabilidade estatística entre os folds.
