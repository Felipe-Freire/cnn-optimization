# Notas

## Encaminhamentos:

Implementar um modelo simplificado de CNN1D voltada para futura implementação em hardware (FPGA), aplicando métodos de otimização de hiperparâmetros.

- Quantização do treinamento CNN. Otimização da alocação de bits por Lagrange e KKT;
- Otimização de Hiperparametros. Busca black-box com penalidade (TPE) e fronteira de Pareto (NSGA-II) - KerasTuner

---

## Objetivos Consolidados e Implementados (`notebooks/01_quantized_cnn1d_optimization.ipynb`)

1. **Redução Arquitetural e Suporte Multi-Resolução (`v0`, `v1`, `v2`)**:
   - Simplificação da topologia CNN1D original para uma arquitetura compacta (`QConv1D` $\to$ `MaxPooling1D` $\to$ `QDense` $\to$ `QDense` de saída) voltada à síntese em FPGA para o sistema de filtragem online do **ATLAS L0 Ringer**;
   - Fatiamento dinâmico por camada de amostragem do calorímetro (`PS, EM1, EM2, EM3, HAD1, HAD2, HAD3`) permitindo alternar entre `v0` (100 anéis), `v1` (50 anéis) e `v2` (25 anéis — $1.376$ MACs totais), preservando a região interna de maior relação sinal-ruído e aplicando normalização $L_1$ por evento.

2. **Alocação Heterogênea Ótima de Bits via Multiplicadores de Lagrange e Condições KKT**:
   - Modelagem da perturbação da função de perda sob quantização uniforme via expansão de Taylor de 2ª ordem ($\Delta \mathcal{L} \approx \sum_i c_i 4^{-b_i}$);
   - Estimativa estocástica livre de matriz do traço da Hessiana $\operatorname{Tr}(\mathbf{H}_i)$ por camada utilizando o **algoritmo de Hutchinson** com vetores de Rademacher;
   - Solução analítica do problema primal-dual via condições **KKT (*water-filling*)** sob orçamento de operações de bits ($\text{BOPs} = \sum_i \text{MAC}_i \cdot b_{a,i} \cdot b_{w,i}$), seguida de treinamento ciente de quantização (*Quantization-Aware Training* — QAT) no **QKeras** (atingindo na topologia `v2` a alocação `[conv1d_1: 3b, dense_1: 6b, output_dense: 8b]` com **$51.776\text{ BOPs}$** — redução de $41.2\%$ frente a 8 bits uniforme).

3. **Validação Estatística ($K$-Fold Cross-Validation + Holdout) Orientada pela Métrica Física ($SP$)**:
   - Monitoramento e *early stopping* guiados diretamente pelo **Índice Soma-Produto ($SP$)** no ponto de joelho (*knee*) da curva ROC;
   - Execução de validação cruzada estratificada ($K=5$ folds) e avaliação final em conjunto *Holdout* isolado ($20\%$), mantendo $SP = 96.48\%$ e probabilidade de detecção $PD = 98.26\%$ em `v2`.

4. **Otimização Multi-Objetivo e Fronteira de Pareto (NSGA-II)**:
   - Exploração conjunta dos hiperparâmetros topológicos (`filters`, `kernel_size`, `dense_units`, `learning_rate`) e larguras de bits (`b_conv`, `b_dense`) via **NSGA-II** para mapear a Fronteira de Pareto entre desempenho físico ($\max SP$) e custo em silício ($\min \text{BOPs}$).

5. **Padronização Hierárquica de Artefatos (Padrão Jubiaba) e Suíte de Visualizações (`codeTCC`)**:
   - Exportação estruturada em `<export_dir>/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/` contendo `saved_models/` (`.keras` por fold e `.onnx`), `plots/`, `history/` (`.csv`/`.json`), `metadata/` (`.pkl`, `.json`, `.csv`) e `evaluation/` (`.npz`);
   - Geração automatizada das visualizações científicas para publicação: perfil médio de energia por camada, curvas de aprendizado, curvas ROC (global, zoom operacional nos *Working Points* do ATLAS e envoltória $\pm 1\sigma$ dos folds), histogramas de scores (linear/log), matriz de confusão no joelho $SP$, boxplots de estabilidade em validação cruzada e Fronteira de Pareto.

