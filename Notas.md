# Notas

## Encaminhamentos:

Implementar um modelo simplificado de CNN1D voltada para futura implementação em hardware (FPGA **Xilinx Virtex UltraScale+ VU13P**), aplicando otimização restrita da alocação de bits por **Multiplicadores de Lagrange e Condições de KKT** em ciclo fechado:

1. **REDE NEURAL QUANTIZADA (QKeras)**: Configura larguras de bits e arquitetura $(w_l, a_l, n_l)$ com *Warm-Start* QAT;
2. **AVALIAÇÃO DE DESEMPENHO E CUSTO**: Calcula a perda pelo Índice-SP ($1 - SP_{\text{val}}$) e estima o consumo físico de recursos na FPGA VU13P (`DSP`, `LUT`, `FF`, `BRAM_bits`);
3. **NÚCLEO DE OTIMIZAÇÃO LAGRANGE / KKT**: Compara o uso com os limites de hardware ($g_j(\mathbf{x}) \le 0$), atualiza os Preços de Sombra ($\mu_j$) e recalcula dinamicamente a precisão ótima $(w_l^*, a_l^*)$ até atingir a convergência por Folga Complementar KKT ($\mu_j^* \cdot g_j(\mathbf{x}^*) = 0$).

---

## Objetivos Consolidados e Implementados (`notebooks/01_quantized_cnn1d_optimization.ipynb`)

1. **Redução Arquitetural e Suporte Multi-Resolução (`v0`, `v1`, `v2`)**:
   - Simplificação da topologia CNN1D original para uma arquitetura compacta (`QConv1D` $\to$ `MaxPooling1D` $\to$ `QDense` $\to$ `QDense` de saída) voltada à síntese em FPGA para o sistema de filtragem online do **ATLAS L0 Ringer**;
   - Fatiamento dinâmico por camada de amostragem do calorímetro (`PS, EM1, EM2, EM3, HAD1, HAD2, HAD3`) permitindo alternar entre `v0` (100 anéis), `v1` (50 anéis) e `v2` (25 anéis — $1.376$ MACs totais), preservando a região interna de maior relação sinal-ruído e aplicando normalização $L_1$ por evento.

2. **Ciclo Fechado Primal-Dual via Multiplicadores de Lagrange e Condições KKT na VU13P**:
   - Estimativa estocástica livre de matriz do traço da Hessiana $\operatorname{Tr}(\mathbf{H}_l)$ por camada utilizando o **algoritmo de Hutchinson** com vetores de Rademacher;
   - Mapeamento físico de recursos para a FPGA **Virtex UltraScale+ VU13P (Tabela 15)** (`DSP: 12.288`, `LUT: 1.728.000`, `FF: 3.456.000`, `BRAM: 94,5 Mb`), avaliando as restrições $g_j(\mathbf{x}) \le 0$;
   - Atualização iterativa dos Preços de Sombra ($\mu_{\text{DSP}}, \mu_{\text{LUT}}, \mu_{\text{FF}}, \mu_{\text{BRAM}}$) acoplada ao passo primal analítico (*water-filling*) e *Warm-Start* QAT no QKeras, atingindo convergência KKT na alocação ótima `w = [3b, 6b, 8b]`, `a = [8b, 6b]` com **$0\text{ DSPs}$**, **$2.364,4\text{ LUTs}$** ($-51,7\%$ frente a 8 bits uniforme) e **$4.824\text{ bits de BRAM}$**.

3. **Validação Estatística ($K$-Fold Cross-Validation + Holdout) Orientada pelo Índice-$SP$ Existente**:
   - Monitoramento, função objetivo primal ($\mathcal{L}_{\text{SP}} = 1 - SP_{\text{val}}$) e *early stopping* guiados diretamente pelo **Índice Soma-Produto ($SP$)** no ponto de joelho (*knee*) da curva ROC;
   - Execução de validação cruzada estratificada ($K=5$ folds) com *Warm-Start* e `ReduceLROnPlateau`, seguida de avaliação final em conjunto *Holdout* isolado ($20\%$).

4. **Padronização Hierárquica de Artefatos (Padrão Jubiaba) e Suíte de Visualizações (`codeTCC` + `VU13P`)**:
   - Exportação estruturada em `<export_dir>/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/` contendo `saved_models/` (`.keras` por fold e `.onnx`), `plots/`, `history/` (`.csv`/`.json`), `metadata/` (`.pkl`, `.json`, `.csv`) e `evaluation/` (`.npz`);
   - Geração automatizada das 10 visualizações científicas em 300 DPI: perfil médio de energia por camada, trajetória primal-dual KKT e preços de sombra ($\mu_j$), ocupação de recursos na VU13P (`LUT`, `FF`, `BRAM`), curvas de aprendizado, curvas ROC (melhor modelo, duelo FP32 vs. KKT e envoltória $\pm 1\sigma$ dos folds), histogramas de scores (linear/log), matriz de confusão no joelho $SP$ e boxplots de estabilidade por *Working Point* do ATLAS.
