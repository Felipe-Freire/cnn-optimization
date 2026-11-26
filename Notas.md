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
   - Subproblema KKT sobre o **envelope convexo inferior contínuo** das curvas separáveis do modelo de custo calibrado na síntese (*water-filling* analítico por partes + projeção inteira guiada por $\boldsymbol{\mu}^*$ com garantia de dualidade fraca `duality_gap >= 0`), acoplado à bisseção externa no orçamento de LUT e *Warm-Start* QAT no QKeras sob teto de 1 % da VU13P. Com 7 pontos calibrados (incluindo `mp_w538` validado fora da amostra), a alocação selecionada é `w = [5b, 3b, 6b]`, `a = [8b, 8b]`, entrada de 8 bits, faixa dos pesos ajustada por camada (`continuous_bits = [4,41; 3,66; 4,51]`, `duality_gap = +0,0067`): **0 DSP**, **9,0 mil LUTs** previstas na calibração (**10,2 mil LUTs** no fold exportado, 68 % do teto de 1 % da VU13P).

3. **Validação Estatística ($K$-Fold Cross-Validation + Holdout) Orientada pelo Índice-$SP$ Existente**:
   - Monitoramento, função objetivo primal ($\mathcal{L}_{\text{SP}} = 1 - SP_{\text{val}}$) e *early stopping* guiados diretamente pelo **Índice Soma-Produto ($SP$)** no ponto de joelho (*knee*) da curva ROC;
   - Validação cruzada estratificada pareada ($K=5$ folds): FP32 de *warm-start* treinado por fold, FP32, 8 bits uniforme e a alocação nos mesmos folds; para `w = [5, 3, 6]`, SP de validação 97,70 ± 0,28 %, diferença pareada alocação − 8 bits de −0,06 ± 0,08 p.p. (erro padrão 0,04) e −0,04 ± 0,04 p.p. frente ao FP32;
   - Holdout isolado ($20\%$) avaliado no limiar escolhido na validação de cada fold (média dos folds): alocação 97,67 %, 8 bits 97,73 %, FP32 97,71 %.
   - Síntese pós-rota de validação fora da amostra (`cnn-fpga-research`, `mp_w538`, `w = [5, 3, 8]` da rodada de 6 pontos, VU13P a 240 MHz): **9.442 LUT** (40 % do 8 bits uniforme sintetizado), **4.848 FF**, **0 DSP**, 1 RAMB18, timing fechado, **11 ciclos (45,8 ns)** contra 12 do 8 bits, II = 1, SP da emulação C++ igual ao do QKeras.

4. **Padronização Hierárquica de Artefatos (Padrão Jubiaba) e Suíte de Visualizações (`codeTCC` + `VU13P`)**:
   - Exportação estruturada em `<export_dir>/<dataset_prefix>/<model_name>/iet{iet}/ieta{ieta}/v{version}/` contendo `saved_models/` (`.keras` por fold e `.onnx`), `plots/`, `history/` (`.csv`/`.json`), `metadata/` (`.pkl`, `.json`, `.csv`) e `evaluation/` (`.npz`);
   - Geração automatizada das 10 visualizações científicas em 300 DPI: perfil médio de energia por camada, trajetória primal-dual KKT e preços de sombra ($\mu_j$), ocupação de recursos na VU13P (`LUT`, `FF`, `BRAM`), curvas de aprendizado, curvas ROC (melhor modelo, duelo FP32 vs. KKT e envoltória $\pm 1\sigma$ dos folds), histogramas de scores (linear/log), matriz de confusão no joelho $SP$ e boxplots de estabilidade por *Working Point* do ATLAS.
