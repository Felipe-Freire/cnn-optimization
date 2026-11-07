# Guia de Estruturação Hierárquica (Padrão Jubiaba), Topologia `v2` (25 Anéis) e Galeria de Visualizações (`codeTCC`)

## 1. Arquitetura de Exportação Hierárquica (Padrão Jubiaba)

Todos os artefatos produzidos pelo pipeline de otimização e treinamento ciente de quantização ([`src/exporter.py`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/exporter.py)) seguem rigorosamente a árvore de diretórios do projeto **Jubiaba**, indexada por prefixo do dataset, nome do modelo, região cinemática (`iet`/`ieta`) e versão da resolução dos anéis (`v0`, `v1`, `v2`):

```text
<export_dir>/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/
├── saved_models/  # Todos os modelos por fold/repeat (model_v{V}_f{F}_r{R}.keras), best_model.keras, best_model.onnx
├── plots/         # Perfis de anéis, curvas de aprendizado, curvas ROC, matriz de confusão, boxplots CV e Pareto
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

### Formulação Matemática do Fatiamento Físico

Para cada camada de amostragem $k \in \{\text{PS}, \text{EM1}, \text{EM2}, \text{EM3}, \text{HAD1}, \text{HAD2}, \text{HAD3}\}$ com número original de anéis $N_k^{(0)}$ e deslocamento inicial $p_k = \sum_{j < k} N_j^{(0)}$, a versão $v \in \{0, 1, 2\}$ com fator de redução $f_v = 2^v$ retém os primeiros $N_k^{(v)} = N_k^{(0)} / f_v$ anéis concêntricos internos:

$$\mathcal{I}_v = \bigcup_{k=1}^{7} \left\{ p_k, \, p_k + 1, \, \dots, \, p_k + \frac{N_k^{(0)}}{2^v} - 1 \right\}$$

Em seguida, aplica-se a normalização $L_1$ por evento sobre o subconjunto selecionado $\mathcal{I}_v$:

$$\hat{x}_{n, i} = \frac{x_{n, i}}{\sum_{j \in \mathcal{I}_v} |x_{n, j}|}, \quad \forall i \in \mathcal{I}_v$$

---

## 3. Resultados Experimentais Consolidados na Topologia `v2` (25 Anéis)

Experimento executado em [`notebooks/01_quantized_cnn1d_optimization.ipynb`](file:///home/iaraujo/Documentos/Github/cnn-optimization/notebooks/01_quantized_cnn1d_optimization.ipynb) e auditado em [`notebooks/02_results_and_visualizations.ipynb`](file:///home/iaraujo/Documentos/Github/cnn-optimization/notebooks/02_results_and_visualizations.ipynb) para a região `iet=5, ieta=0` (`77.126` eventos balanceados: `61.700` em desenvolvimento 5-Fold CV e `15.426` em Holdout isolado):

| Modelo | Topologia | Precisão por Camada (KKT) | SP Holdout | PD Holdout | FA Holdout | MACs | BOPs | Economia de BOPs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline FP32** | `v2` (25 anéis) | Float32 | $97.39\%$ | $98.11\%$ | $3.33\%$ | $1.376$ | $88.064$ (eq. 8b) | Ref. |
| **QCNN1D KKT (Melhor Fold #4)** | `v2` (25 anéis) | `conv1d_1`: **3b**, `dense_1`: **6b**, `output_dense`: **8b** | **$96.48\%$** | **$98.26\%$** | **$5.28\%$** | **$1.376$** | **$51.776$** | **$-41.2\%$** |

*Comparado à topologia `v1` (50 anéis, $113.472$ BOPs com KKT), a topologia `v2` (25 anéis, $51.776$ BOPs com KKT) proporciona uma redução adicional de **$54.4\%$ em BOPs** e **$51.0\%$ em MACs** ($1.376$ vs. $2.808$), mantendo probabilidade de detecção $PD = 98.26\%$ no conjunto Holdout.*

---

## 4. Catálogo de Visualizações (`src/visualization.py`)

O módulo [`src/visualization.py`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py) implementa todas as visualizações clássicas de `codeTCC` e da metodologia KKT/Pareto:

1. [`plot_rings_profile`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L26-L85) (`rings_profile_v2.png`): Perfil médio e desvio padrão da energia normalizada para Elétrons vs. Jatos com linhas divisórias das 7 camadas do calorímetro;
2. [`plot_learning_curves`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L88-L144) (`learning_curves.png`): Curvas de aprendizado de Loss (BCE), Acurácia e evolução das métricas do trigger ($SP, PD, 1-FA$);
3. [`plot_best_model_roc`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L147-L211) (`roc_best_model.png`): Curva ROC global e curva ROC com zoom na região de operação do trigger ($PD \in [94\%, 100\%]$) destacando os *Working Points* `Tight (96%)`, `Medium (97.5%)` e `Loose (98.5%)`;
4. [`plot_cv_roc_envelope`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L214-L259) (`roc_cv_envelope.png`): Curva ROC média e banda de incerteza $\pm 1\sigma$ sobre os 5 folds de validação cruzada;
5. [`plot_output_score_histogram`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L262-L300) (`output_scores_hist.png`): Histograma de densidade das saídas da rede (Sinal vs. Fundo) em escalas linear e logarítmica com marcação do limiar ótimo do joelho $SP$;
6. [`plot_confusion_matrix_knee`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L303-L347) (`confusion_matrix_knee.png`): Matriz de confusão percentual e absoluta avaliada exatamente no limiar do joelho $SP$;
7. [`plot_cv_statistics_boxplots`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L350-L384) (`cv_boxplots.png`): Boxplots de distribuição de $SP_{\max}$, $PD_{\text{knee}}$ e $FA_{\text{knee}}$ entre os folds;
8. [`plot_fa_per_fold_workpoints`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L387-L431) (`fa_per_fold_workpoints.png`): Boxplots da taxa de Falso Alarme ($FA$) por fold fixando os pontos de operação do ATLAS (`Tight`, `Medium`, `Loose`, `Very Loose`);
9. [`plot_pareto_and_roc_duel`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/visualization.py#L434-L499) (`pareto_and_roc_duel.png`): Duelo direto da curva ROC (FP32 vs. QKeras KKT) e Fronteira de Pareto multi-objetivo gerada via NSGA-II.
