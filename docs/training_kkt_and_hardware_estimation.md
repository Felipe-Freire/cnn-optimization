# Guia Completo do Pipeline: Processo de Treinamento, Otimização de Bits (Lagrange/KKT) e Estimação de Recursos na FPGA VU13P

Este documento detalha de ponta a ponta como o projeto executa o **treinamento da rede neural convolucional 1D (`QCNN1D`)**, a **otimização automática da largura de bits por camada via Multiplicadores de Lagrange e Condições de Karush-Kuhn-Tucker (KKT)** e a **estimação de recursos físicos na FPGA AMD/Xilinx Virtex UltraScale+ VU13P** com integração ao `qkeras.qtools`.

---

## Visão Geral do Ciclo Fechado em 3 Etapas

Todo o fluxo implementado em [`notebooks/01_quantized_cnn1d_optimization.ipynb`](file:///home/iaraujo/Documentos/Github/cnn-optimization/notebooks/01_quantized_cnn1d_optimization.ipynb) opera sobre um ciclo fechado de três estágios acoplados:

```mermaid
flowchart TD
    D["Dados ATLAS L0 Ringer (v2 = 25 Anéis)<br/>Normalização L1 + Split Estratificado (80% Dev / 20% Holdout)"] --> R["Treinamento do Modelo Referência FP32<br/>+ Cálculo do Traço da Hessiana de Hutchinson Tr(H_l)"]
    R --> S1

    subgraph LOOP["Ciclo Fechado Primal-Dual (Iterações t = 1 ... T)"]
        S1["1. REDE NEURAL QUANTIZADA (QKeras)<br/>Instancia QConv1D e QDense com bits (w_l, a_l)<br/>+ Warm-Start dos pesos FP32 + Fine-Tuning QAT"]
        S2["2. AVALIAÇÃO DE DESEMPENHO E CUSTO<br/>• Calcula SP_val no joelho da ROC e perda (1 - SP_val)<br/>• Estima recursos na VU13P (DSP, LUT, FF, BRAM_bits) + QTools"]
        S3["3. NÚCLEO DE OTIMIZAÇÃO LAGRANGE / KKT<br/>• Avalia restrições g_j(x) = R_j(x) - B_j ≤ 0<br/>• Atualiza Preços de Sombra (μ_j) via subgradiente dual<br/>• Calcula novos bits ótimos (w_l*, a_l*) via Water-Filling KKT"]
        S1 --> S2 --> S3
        S3 -- "Não convergiu (t → t+1)" --> S1
    end

    S3 -- "Convergência KKT Atingida<br/>(SP estável + g_j ≤ 0 + μ_j·g_j = 0)" --> CV["Treinamento Final K-Fold (5 Folds) com Warm-Start<br/>+ Avaliação no Holdout Isolado + Exportação Jubiaba"]
```

---

## Parte 1: O Processo de Pré-Processamento e Treinamento

### 1.1 Fatiamento Físico da Topologia `v2` (25 Anéis) e Normalização $L_1$
Os sinais de entrada provêm dos calorímetros eletromagnético (PS, EM1, EM2, EM3) e hadrônico (HAD1, HAD2, HAD3) do experimento ATLAS no LHC, na região cinemática $(\text{iet}=5, \text{ieta}=0)$, contendo um conjunto balanceado de **77.126 eventos** ($38.563$ elétrons de sinal com $y=1$ e $38.563$ jatos hadrônicos de ruído de fundo com $y=0$).

1. **Seleção dos 25 Anéis (`v2`)**: A função [`slice_rings_by_version(X_raw, version="v2")`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/metrics.py#L43-L81) extrai os primeiros anéis concêntricos mais internos de cada camada de amostragem a partir do vetor original de 100 anéis (`v0`):
   - `PS`: 2 anéis | `EM1`: 16 anéis | `EM2`: 2 anéis | `EM3`: 2 anéis | `HAD1`: 1 anel | `HAD2`: 1 anel | `HAD3`: 1 anel $\implies \mathbf{M = 25\text{ anéis}}$.
2. **Normalização $L_1$ por Evento**: A função [`norm1(X)`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/metrics.py#L84-L90) normaliza a energia depositada em cada anel pela energia absoluta total do respectivo chuveiro, garantindo invariância de escala frente à energia transversal ($E_T$):
   $$\hat{x}_i = \frac{x_i}{\sum_{m=1}^{25} |x_m|}, \quad i = 1, \dots, 25$$
3. **Particionamento Estratificado**:
   - **Conjunto de Desenvolvimento (`X_dev`, 80% = 61.700 eventos)**: utilizado para o treinamento de referência FP32, o loop de alocação de bits KKT e a validação cruzada em 5 folds;
   - **Conjunto de Teste Isolado (`X_holdout`, 20% = 15.426 eventos)**: mantido completamente intocado durante todo o treinamento e otimização, servindo exclusivamente para o laudo final de generalização.

---

### 1.2 Arquitetura da Rede Neural (`Compact_CNN1D_FP32` e `Compact_QCNN1D`)
Definida em [`src/models.py`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/models.py#L12-L126), a topologia é uma CNN 1D ultracompacta projetada para caber nos limites estritos de área e latência do *trigger* de Nível 0 (L0):

| Ordem | Camada (`name`) | Tipo (FP32 / QKeras) | Formato de Saída (`v2`) | Hiperparâmetros e Quantizadores | Parâmetros ($P_l$) | Operações ($\text{MAC}_l$) |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: |
| 1 | `input_rings` / `reshape_1d` | `Input` + `Reshape` | $(25, 1)$ | Entrada normalizada $L_1$ | $0$ | $0$ |
| 2 | `conv1d_1` | `Conv1D` / `QConv1D` | $(25, 8)$ | $n_1 = 8$ filtros, $K = 3$, `padding="same"`, `quantized_bits(w_1, int_w1, alpha=1)` | $32$ | $600$ |
| 3 | `relu_conv1` | `ReLU` / `QActivation` | $(25, 8)$ | `quantized_relu(a_1, int_a1)` | $0$ | $0$ |
| 4 | `maxpool_1` + `flatten_1` | `MaxPooling1D(2)` + `Flatten` | $(96,)$ | Redução espacial $25 \to 12$ ($12 \times 8 = 96$) | $0$ | $0$ |
| 5 | `dense_1` | `Dense` / `QDense` | $(8,)$ | $n_2 = 8$ neurônios, `quantized_bits(w_2, int_w2, alpha=1)` | $776$ | $768$ |
| 6 | `relu_dense1` | `ReLU` / `QActivation` | $(8,)$ | `quantized_relu(a_2, int_a2)` | $0$ | $0$ |
| 7 | `output_dense` | `Dense` / `QDense` | $(1,)$ | $1$ neurônio, `quantized_bits(w_3, int_w3, alpha=1)` + `sigmoid` | $9$ | $8$ |
| **Total** | — | — | — | — | **$817$** | **$1.376$** |

---

### 1.3 Função de Perda Diferenciável vs. Critério Físico de Seleção (`SPCallback`)
Durante o treinamento supervisionado no TensorFlow/QKeras, utilizamos **duas métricas complementares**:

1. **Para o Cálculo dos Gradientes (*Backpropagation*) e `ReduceLROnPlateau`**:
   - Utiliza-se a **Entropia Cruzada Binária (`binary_crossentropy` / `val_loss`)** com o otimizador **Adam**. Como a entropia cruzada é suave e diferenciável em todos os pontos, ela fornece gradientes estáveis para atualizar os pesos (via *Straight-Through Estimator* - STE nos quantizadores QKeras) e orienta o agendador `ReduceLROnPlateau` (que reduz a taxa de aprendizado pela metade quando a `val_loss` estaciona por 3 épocas).
2. **Para a Avaliação Física, Seleção da Melhor Época e Parada Antecipada (`SPCallback`)**:
   - No sistema de *trigger* do ATLAS, é essencial maximizar simultaneamente a **Probabilidade de Detecção de Elétrons ($PD = TPR$)** e a **Rejeição de Jatos de Fundo ($1 - FA = 1 - FPR$)**.
   - Ao final de cada época, o callback [`SPCallback`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/metrics.py#L140-L216) varre todos os limiares possíveis $\tau \in [0, 1]$ da curva ROC de validação através de [`compute_sp_with_threshold()`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/metrics.py#L106-L137) e encontra o ponto de operação ótimo ("joelho" ou *knee* da ROC) que maximiza o **Índice Soma-Produto ($SP$)**:
     $$SP(\tau) = \sqrt{\sqrt{PD(\tau) \cdot (1 - FA(\tau))} \cdot \left[ \frac{PD(\tau) + (1 - FA(\tau))}{2} \right]}, \qquad SP_{\text{val}} = \max_{\tau \in [0, 1]} SP(\tau)$$
   - O `SPCallback` salva em memória os pesos da época que atingiu o maior $SP_{\text{val}}$ (`restore_best_weights()`) e interrompe o treinamento caso o $SP_{\text{val}}$ não supere a marca histórica por `patience` épocas consecutivas.

### 1.4 Transferência de Pesos (*Warm-Start* QAT)
A função [`transfer_weights_to_qmodel(model_fp32, qmodel)`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/models.py#L129-L146) copia os pesos já convergidos do modelo de referência FP32 para o modelo QKeras antes de iniciar o *Quantization-Aware Training* (QAT), eliminando variância estocástica de inicialização e acelerando a convergência. Na validação cruzada, o FP32 de cada fold é treinado só com o treino daquele fold (`src/training.py`), para que nenhum evento de validação tenha sido visto pelo *warm-start*; FP32, 8 bits uniforme e a alocação escolhida usam os mesmos folds, a mesma semente e o mesmo *warm-start*, e o holdout é avaliado no limiar escolhido na validação de cada fold.

O QAT usa taxa de aprendizado 3e-4 (menor que a do FP32), faixa de cada quantizador de pesos ajustada a max|W| do FP32 de *warm-start* e checkpoint de maior SP **entre as épocas com `val_loss` ≤ 1,2× a mínima do treino** (`SPCallback(max_loss_ratio=1.2)`). Como o SP só depende da ordenação, escolher a época apenas pelo SP às vezes guardava um modelo descalibrado (`val_loss` dezenas de vezes a mínima, limiar do joelho de 1e-5 ou menos, que a sigmoide em ponto fixo do FPGA não representa); em 3 folds × 5 configurações de poucos bits isso acontecia em 7 a 8 de 15 treinos e deixou de acontecer com a regra.

---

## Parte 2: Como Funciona a Otimização da Quantidade de Bits (Lagrange e KKT)

O problema central consiste em escolher o vetor de larguras de bits de pesos $\mathbf{w} = [w_1, w_2, w_3]^T$ e ativações $\mathbf{a} = [a_1, a_2]^T$ que minimiza a perda de desempenho físico $\mathcal{L}_{\text{SP}}(\mathbf{x}) = 1 - SP_{\text{val}}(\mathbf{x})$ respeitando simultaneamente os limites máximos de recursos $B_j$ alocados à instância do classificador na FPGA **Virtex UltraScale+ VU13P**:

$$\min_{\mathbf{x} = (\mathbf{w}, \mathbf{a})} \mathcal{L}_{\text{SP}}(\mathbf{x}) = 1 - SP_{\text{val}}(\mathbf{x}) \quad \text{sujeito a} \quad g_j(\mathbf{x}) = R_j(\mathbf{x}) - B_j \le 0, \quad \forall j \in \{\text{DSP}, \text{LUT}, \text{FF}, \text{BRAM\_bits}\}$$

A resolução implementada em [`src/kkt_allocator.py`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/kkt_allocator.py) combina **Sensibilidade Hessiana de 2ª Ordem** (nível intra-camada) com **Ascensão Dual de Multiplicadores de Lagrange** (nível de recursos da VU13P):

1. **Sensibilidade de Curvatura via Algoritmo de Hutchinson ($\operatorname{Tr}(\mathbf{H}_l)$ e $c_l$)**:
   - [`compute_hutchinson_hessian_trace`](file:///home/felipe/cern/cnn-optimization/src/kkt_allocator.py) estima o traço total da Hessiana de cada camada usando produtos Hessiana-vetor (HVP) com vetores de Rademacher $\mathbf{z} \in \{-1, +1\}^{P_l}$, sorteados camada a camada, e [`compute_layer_c_coefficients`](file:///home/felipe/cern/cnn-optimization/src/kkt_allocator.py) computa $c_l = \operatorname{Tr}(\mathbf{H}_l) \, \kappa_l^2 / 24$, com $\kappa_l = 2^{I_l+1}$ a largura da faixa ajustada por camada do quantizador de pesos.
2. **Envelope Convexo Contínuo e Passo Primal-Dual KKT (`solve_relaxed_dual_envelope`)**:
   - Em [`LagrangeKKTQuantizerOptimizer.compute_primal_bit_update`](file:///home/felipe/cern/cnn-optimization/src/kkt_allocator.py), para cada camada $l$ e recurso $j$, a curva separável de custo do modelo calibrado $f_{jl}(w)$ é substituída pelo seu **envelope convexo inferior** linear por partes $\hat{f}_{jl}(w) \le f_{jl}(w)$ sobre $[b_{\min}, w_{\max}]$, formando uma relaxação estritamente convexa válida.
   - Com $B_{\text{DSP}} = 0$, limita $w_l, a_l \le 8\text{ bits}$ (sem `DSP48E2`). Para preços $\boldsymbol{\mu} \ge 0$, o passo primal resolve o *water-filling* analítico por partes (`_piecewise_water_filling`) satisfazendo a estacionaridade por subdiferencial $0 \in \partial_{w_l}\mathcal{L}$, enquanto o passo dual ajusta $\mu_j$ em escala logarítmica até a viabilidade primal e a folga complementar exatas ($\mu_j^* \hat{g}_j(\mathbf{w}^*) = 0$).
3. **Projeção Inteira Guiada pelo KKT e Bisseção Externa em $B_{\text{LUT}}$**:
   - Os bits inteiros derivam diretamente da solução contínua $\mathbf{w}^*(\boldsymbol{\mu}^*)$ (piso + reparo de viabilidade + preenchimento guloso ponderado por $\boldsymbol{\mu}^*$ + refinamento local por trocas $O(L^2)$ sob o modelo calibrado real), com garantia de dualidade fraca ($\Omega(\mathbf{w}_{\text{int}}) - q(\boldsymbol{\mu}^*) \ge 0$).
   - O notebook procura, por bisseção no orçamento de LUT, o menor custo que preserva o SP do 8 bits uniforme (seção 5 de `theory_and_formulation.md`). Em `et5_eta0` (teto de 1 % da VU13P), a escolha da rodada com 6 pontos de calibração foi `w = [5, 3, 8]`, sintetizada fora da amostra no `cnn-fpga-research` (`mp_w538`; hls4ml → Vitis HLS → Vivado, VU13P, 240 MHz, pós-rota): 9.442 LUT, 4.848 FF, 0 DSP, 1 RAMB18, timing fechado (folga de 0,60 ns), latência de 11 ciclos (45,8 ns, contra 12 do 8 bits uniforme) e II = 1, com emulação C++ bit a bit igual ao QKeras no holdout (97,85 %). Com `mp_w538` incorporado como 7º ponto no domínio (e filtro de alavancagem `< 0,9` nas margens LOO) e o solver KKT no envelope convexo contínuo, a alocação confirmada é `w = [5, 3, 6]`, `a = [8, 8]`, entrada de 8 bits (`continuous_bits = [4,41; 3,66; 4,51]`, `duality_gap = +0,0067`, resíduo $3{,}5\times 10^{-18}$): 9,0 mil LUTs na calibração (10,2 mil LUTs e 3,9 mil FFs no fold exportado, 0 DSP). CV pareada em 5 folds: SP 97,70 ± 0,28 %, diferença para o 8 bits uniforme de −0,06 ± 0,08 p.p. (erro padrão 0,04) e para o FP32 de −0,04 ± 0,04 p.p.; holdout no limiar da validação, média dos folds: 97,67 % (8 bits: 97,73 %, FP32: 97,71 %).

---

## Parte 3: Estimação de Recursos na VU13P e Integração com `qkeras.qtools`

Em [`src/hardware.py`](file:///home/iaraujo/Documentos/Github/cnn-optimization/src/hardware.py), a estimação combina o modelo aritmético de síntese UltraScale+ com a inspeção direta do grafo quantizado via **`qkeras.qtools`**:

1. **Extração Exata de Tipos `ap_fixed<W, I>` e Energia via `qkeras.qtools` (`extract_qtools_hardware_profile`)**:
   - Infere automaticamente a largura real dos **multiplicadores (`mult_bits`)** e **acumuladores (`acc_bits`, `acc_int_bits`)** necessária para evitar *overflow* em hardware:
     - `conv1d_1` ($w_1=2, a_1=8$): multiplicador `ap_fixed<10, 2>`, acumulador **`ap_fixed<12, 4>`**;
     - `dense_1` ($w_2=5, a_2=8$): multiplicador `ap_fixed<13, 5>`, acumulador **`ap_fixed<20, 12>`**;
     - `output_dense` ($w_3=8, a_3=8$): multiplicador `ap_fixed<16, 5>`, acumulador **`ap_fixed<20, 9>`**;
   - Calcula o consumo energético por inferência em **picojoules (pJ)** via `qtools.pe()` ($240\text{ pJ}$ de núcleo computacional / $271\text{ pJ}$ incluindo I/O para o modelo ótimo KKT).
2. **Recursos da VU13P pelo modelo calibrado (`calibrated_resources`, `make_cost_fn`)**:
   - Conta somadores das árvores de soma a partir dos pesos inteiros (zeros e potências de 2 não custam multiplicador; constantes com $d$ dígitos CSD custam $d - 1$ somadores; constantes largas vão para DSP48E2), mais a lógica de ReLU/pooling; `BRAM_bits` é a tabela da sigmoide (1 RAMB18);
   - Coeficientes ajustados nas sínteses desta rede (`src/calibration/vu13p_hls4ml_synthesis.json`); as margens do ajuste entram no orçamento (`get_vu13p_budget(..., margins=...)`). Detalhes na Seção 4 de `theory_and_formulation.md`;
   - `estimate_vu13p_detailed_resources` continua fornecendo tipos de acumulador e energia do QTools; sua latência é uma estimativa analítica (as sínteses deram 9 a 12 ciclos a 240 MHz, II = 1).
