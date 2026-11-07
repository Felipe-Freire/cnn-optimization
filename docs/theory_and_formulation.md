# Fundamentação Teórica e Formulação Matemática

Este documento consolida a modelagem matemática, deduções analíticas e critérios de projeto para a otimização de Redes Neurais Convolucionais 1D (CNN1D) quantizadas via QKeras para implementação em hardware (FPGA), no contexto do sistema de filtragem de elétrons do ATLAS L0 Ringer (HL-LHC), alinhado às propostas de otimização de hiperparâmetros e quantização baseada em sensibilidade hessiana.

---

## 1. Contexto Físico e Métrica de Eficiência (ATLAS Ringer)

O calorímetro do experimento ATLAS no LHC é segmentado em camadas de amostragem cilíndricas. A técnica Ringer compacta a informação espacial dos chuveiros eletromagnéticos e hadrônicos em anéis concêntricos em torno do baricentro de deposição de energia.

A normalização de energia por evento garante invariância de escala em relação à energia transversal ($E_T$):

$$\hat{x}_i = \frac{x_i}{\sum_{j=1}^{M} |x_j|}, \quad i = 1, \dots, M$$

onde $M$ é o número de anéis (100 na versão completa `v0`, 50 na versão reduzida `v1`, 25 na versão `v2`).

### 1.1 Índice Soma-Produto (SP)

A discriminação entre eventos de interesse (elétrons, sinal $y=1$) e ruído de fundo (jatos hadrônicos, fundo $y=0$) é avaliada pela curva ROC (*Receiver Operating Characteristic*), parametrizada pela Probabilidade de Detecção ($PD$, *True Positive Rate*) e Probabilidade de Alarme Falso ($FA$, *False Positive Rate*):

$$PD = \frac{TP}{TP + FN}, \qquad FA = \frac{FP}{FP + TN}$$

O ponto operacional ótimo é definido no "joelho" (*knee*) da curva ROC através da maximização do **Índice Soma-Produto ($SP$)**:

$$SP(PD, FA) = \sqrt{\sqrt{PD \cdot (1 - FA)} \cdot \left[ \frac{PD + (1 - FA)}{2} \right]}$$

Como o treinamento estocástico introduz variância, a métrica de validação cruzada avaliada em $K$ folds é:

$$SP(\boldsymbol{\theta}) = \frac{1}{K} \sum_{k=1}^K SP_k(\boldsymbol{\theta})$$

---

## 2. Modelagem do Custo de Hardware em FPGA

Em uma arquitetura de hardware dedicado (ex.: AMD Xilinx UltraScale+ FPGA), os principais recursos limitantes para redes neurais são:
1. **Multiplicadores em Blocos de DSP (Digital Signal Processors)**;
2. **Tabelas de Consulta (LUTs - Look-Up Tables)** para lógica distribuída e aritmética;
3. **Memória de Bloco (BRAM/URAM)** e registradores para *line buffers* e armazenamento de pesos.

### 2.1 Multiplicações e Acumulações (MACs)

Para uma camada convolucional 1D e uma camada densa:

$$\text{MAC}_{\text{conv}, \ell} = L_{\ell}^{\text{out}} \cdot K_{\ell} \cdot C_{\ell}^{\text{in}} \cdot C_{\ell}^{\text{out}}$$

$$\text{MAC}_{\text{dense}, \ell} = n_{\ell}^{\text{in}} \cdot n_{\ell}^{\text{out}}$$

onde $L^{\text{out}}$ é o comprimento temporal da sequência na saída, $K$ é o tamanho do kernel, e $C^{\text{in}}, C^{\text{out}}$ são os canais de entrada e saída.

### 2.2 Custo de Operações em Bits (BOPs - Bit Operations)

Para pesos com $b_{w,i}$ bits e ativações de entrada com $b_{a,i}$ bits:

$$\text{BOPs} = \sum_{i} a_i b_{w,i}, \quad \text{onde } a_i = \text{MAC}_i \cdot b_{a,i}$$

Quando os bits de ativação são fixados (ex.: 8 bits), os $\text{BOPs}$ tornam-se lineares nos bits dos pesos $b_{w,i}$.

---

## 3. Otimização de Precisão Mista por Multiplicadores de Lagrange e KKT

Em vez de forçar quantização uniforme ($b_i = 8$ bits em todas as camadas), a precisão mista aloca mais bits para camadas críticas e menos bits para camadas com alto custo de hardware.

### 3.1 Modelo da Perturbação e Sensibilidade Hessiana

Um quantizador uniforme simétrico de $b_i$ bits sobre o intervalo $[-r_i, r_i)$ possui passo de quantização:

$$\Delta_i = \frac{2 r_i}{2^{b_i}} = 2 r_i \cdot 2^{-b_i}$$

Sob a hipótese de ruído de arredondamento uniforme, a variância do erro de quantização de cada peso é $\sigma^2 = \frac{\Delta_i^2}{12}$. Para os $n_i$ parâmetros da camada $i$:

$$\|\mathbf{W}_{q,i} - \mathbf{W}_i\|_2^2 \approx n_i \frac{\Delta_i^2}{12} = \frac{n_i r_i^2}{3} 2^{-2 b_i} = \frac{n_i r_i^2}{3} 4^{-b_i}$$

Pela expansão de Taylor de segunda ordem da função de perda $\mathcal{L}$ em torno do ponto de mínimo ($\nabla_{\mathbf{W}} \mathcal{L} \approx \mathbf{0}$):

$$\Delta \mathcal{L} \approx \frac{1}{2} \Delta \mathbf{W}^T \mathbf{H} \, \Delta \mathbf{W} \approx \sum_i \Omega_i(b_i)$$

$$\Omega_i(b_i) \approx c_i 4^{-b_i}, \quad \text{onde } c_i = \operatorname{Tr}(\mathbf{H}_i) \frac{n_i r_i^2}{3}$$

#### Estimativa do Traço da Hessiana via Algoritmo de Hutchinson (HVP)
Como calcular a matriz Hessiana completa $\mathbf{H} \in \mathbb{R}^{P \times P}$ é inviável, o traço $\operatorname{Tr}(\mathbf{H}_i)$ é estimado sem formar a matriz através de produtos Hessiana-vetor (HVP) com vetores aleatórios de Rademacher $\mathbf{z} \in \{-1, +1\}^P$:

$$\operatorname{Tr}(\mathbf{H}_i) \approx \frac{1}{M_H} \sum_{j=1}^{M_H} \mathbf{z}_j^T \left( \nabla^2_{\mathbf{W}_i} \mathcal{L} \, \mathbf{z}_j \right) = \frac{1}{M_H} \sum_{j=1}^{M_H} \mathbf{z}_j^T \nabla_{\mathbf{W}_i} \left( \nabla_{\mathbf{W}_i} \mathcal{L}^T \mathbf{z}_j \right)$$

O cálculo de cada HVP requer apenas duas passagens de gradiente por diferenciação automática, sem custo quadrático de memória.

---

### 3.2 O Problema de Otimização Primal

Dado o orçamento máximo de BOPs $B$ e limites físicos discretos $b_{\min} \le b_i \le b_{\max}$:

$$\min_{\mathbf{b}} \Omega(\mathbf{b}) = \sum_i c_i 4^{-b_i}$$

$$\text{sujeito a:} \quad g(\mathbf{b}) = \sum_i a_i b_i - B \le 0$$

$$b_{\min} \le b_i \le b_{\max}, \quad \forall i$$

#### Convexidade Estrita e Condição de Slater
A segunda derivada da função objetivo em relação a cada $b_i$ é:

$$\frac{d^2}{d b_i^2} (c_i 4^{-b_i}) = c_i (\ln 4)^2 4^{-b_i} > 0 \quad (\text{para } c_i > 0)$$

Portanto, $\Omega(\mathbf{b})$ é estritamente convexa. Como as restrições são lineares afins e existe um ponto estritamente viável ($\sum a_i b_{\min} < B$, condição de Slater), **as condições de Karush-Kuhn-Tucker (KKT) são necessárias e suficientes**: a solução que satisfaz KKT é o mínimo global único.

---

### 3.3 Formulação do Lagrangiano e Condições KKT

Com multiplicador $\lambda \ge 0$ para a restrição de orçamento e $\nu_i^-, \nu_i^+ \ge 0$ para os limites de caixa:

$$\mathcal{L}(\mathbf{b}, \lambda, \boldsymbol{\nu}^-, \boldsymbol{\nu}^+) = \sum_i c_i 4^{-b_i} + \lambda \left( \sum_i a_i b_i - B \right) + \sum_i \nu_i^- (b_{\min} - b_i) + \sum_i \nu_i^+ (b_i - b_{\max})$$

#### Sistema KKT:
1. **Estacionaridade**:
   $$\frac{\partial \mathcal{L}}{\partial b_i} = -c_i \ln 4 \cdot 4^{-b_i} + \lambda a_i - \nu_i^- + \nu_i^+ = 0, \quad \forall i$$

2. **Viabilidade Primal**:
   $$\sum_i a_i b_i \le B, \qquad b_{\min} \le b_i \le b_{\max}, \quad \forall i$$

3. **Viabilidade Dual**:
   $$\lambda \ge 0, \qquad \nu_i^- \ge 0, \qquad \nu_i^+ \ge 0, \quad \forall i$$

4. **Folga Complementar**:
   $$\lambda \left( \sum_i a_i b_i - B \right) = 0$$
   $$\nu_i^- (b_{\min} - b_i) = 0, \qquad \nu_i^+ (b_i - b_{\max}) = 0, \quad \forall i$$

---

### 3.4 Solução Analítica (Water-Filling)

Para camadas no interior do domínio viável ($\nu_i^- = 0, \nu_i^+ = 0$):

$$-c_i \ln 4 \cdot 4^{-b_i} + \lambda a_i = 0 \implies 4^{-b_i} = \frac{\lambda a_i}{c_i \ln 4} \implies -2 b_i = \log_2 \left( \frac{\lambda a_i}{c_i \ln 4} \right)$$

$$b_i^*(\lambda) = \frac{1}{2} \log_2 \left( \frac{c_i \ln 4}{\lambda a_i} \right)$$

Com a projeção nos limites de caixa $[b_{\min}, b_{\max}]$:

$$b_i^*(\lambda) = \min\left( b_{\max}, \max\left( b_{\min}, \frac{1}{2} \log_2 \left( \frac{c_i \ln 4}{\lambda a_i} \right) \right) \right)$$

#### Resolução do Problema Dual
A função de folga do orçamento:

$$h(\lambda) = \sum_i a_i b_i^*(\lambda) - B$$

é monótona não crescente em $\lambda$. 
- Se $\sum_i a_i b_{\max} \le B$, o orçamento é inativo, $\lambda = 0$ e todas as camadas recebem $b_{\max}$.
- Caso contrário, $\lambda^* > 0$ é a raiz única de $h(\lambda) = 0$, encontrada via:
  1. **Método da Bisseção** (convergência global robusta e garantida);
  2. **Método de Newton** na variável $t = \log_2 \lambda$ para aceleração quadrática local.

#### Interpretação Física e de Otimização:
- Camadas com maior sensibilidade relativa $\frac{c_i}{a_i}$ recebem mais bits de precisão;
- Camadas com grande volume de computação $a_i$ têm bits penalizados pelo custo;
- $\lambda^* = -\frac{d\Omega^*}{dB}$ é o **preço-sombra** (*shadow price*), medindo a taxa marginal de redução da perda por unidade extra de orçamento de hardware.

---

## 4. Otimização Bi-Objetivo de Hiperparâmetros (Pareto & TPE)

### 4.1 Abordagem 1: Busca Black-Box com Penalidade Exterior (TPE)

A função objetivo escalarizada incorpora o orçamento de hardware como penalidade suave:

$$\min_{\boldsymbol{\theta}} \Phi_\mu(\boldsymbol{\theta}) = -SP(\boldsymbol{\theta}) + \mu \left[ \max\left(0, \frac{\text{BOPs}(\boldsymbol{\theta}) - B}{B}\right) \right]^2$$

O estimador TPE (*Tree-structured Parzen Estimator*) constrói distribuições probabilísticas $p(\boldsymbol{\theta}|y)$ para guiar a busca de filtros, tamanhos de kernel e taxas de aprendizado em direção a regiões viáveis de alto índice $SP$.

### 4.2 Abordagem 2: Fronteira de Pareto com Algoritmo Genético NSGA-II

Na formulação bi-objetivo:

$$\max_{\boldsymbol{\theta}} f_1(\boldsymbol{\theta}) = SP(\boldsymbol{\theta})$$

$$\min_{\boldsymbol{\theta}} f_2(\boldsymbol{\theta}) = \text{BOPs}(\boldsymbol{\theta})$$

O algoritmo **NSGA-II** identifica o conjunto de soluções não dominadas $\mathcal{P}^*$, gerando a curva de *trade-off* explícita entre física e recursos de hardware para escolha do ponto ótimo de síntese em FPGA.
