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

## 4. Formulação Primal-Dual Restrita para o FPGA Xilinx Virtex UltraScale+ VU13P

No contexto real de operação do trigger de Nível 0 (L0) do experimento ATLAS no HL-LHC, as placas de processamento de sinais (como a plataforma Global Common Module - GCM) utilizam FPGAs da família **AMD/Xilinx Virtex UltraScale+ VU13P**.

Conforme a **Tabela 15 (*Virtex UltraScale+ FPGA Feature Summary*)** da documentação oficial da AMD/Xilinx, a capacidade total do dispositivo ($\mathbf{B}_{\text{chip}}$) e o orçamento fracionário dedicado a uma instância individual do classificador Ringer ($B_j = \alpha_j \cdot B_{j, \text{chip}}$, onde $\alpha_j \in (0, 1]$ representa a fração de área/recursos alocada ao núcleo neural) são definidos por:

| Recurso Físico ($j$) | Capacidade Total da VU13P ($B_{j, \text{chip}}$) | Modelo de Consumo por Camada $l \in \{1, \dots, L\}$ |
| :--- | :--- | :--- |
| **DSP Slices** (`DSP`) | $12.288\text{ slices}$ | $\text{DSP}_l(w_l, a_l, n_l)$ — Alocado quando $w_l > 8$ ou $a_l > 8$ |
| **CLB LUTs** (`LUT`) | $1.728.000\text{ LUTs}$ | $\text{LUT}_l(w_l, a_l, n_l) \approx \text{MAC}_l \cdot \frac{w_l \cdot a_l}{18}$ (quando $w_l \le 8, a_l \le 8$) |
| **CLB Flip-Flops** (`FF`) | $3.456.000\text{ FFs}$ | $\text{FF}_l(w_l, a_l, n_l) \approx P_l \cdot w_l + N_{\text{out}, l} \cdot a_l$ |
| **Block RAM** (`BRAM`) | $94,5\text{ Mb}$ ($2.688 \times 36\text{ Kb}$) | $\text{Bits}_l(w_l, n_l) = P_l \cdot w_l$ |
| **UltraRAM** (`URAM`) | $360,0\text{ Mb}$ ($1.280 \times 288\text{ Kb}$) | Memória massiva unificada ($B_{\text{Mem}} = B_{\text{BRAM}} + B_{\text{URAM}}$) |

### 4.1 Variáveis de Decisão e Função Objetivo Baseada no Índice-SP Existente

Para cada camada $l \in \{1, \dots, L\}$ da arquitetura `QConv1D` / `QDense`:
- $w_l \in \{1, 2, 4, 6, 8, 16\}$: Largura de bits dos pesos e vieses (*weights/biases*);
- $a_l \in \{1, 2, 4, 6, 8, 16\}$: Largura de bits das ativações (*activations*);
- $n_l \in \mathbb{Z}^+$: Número de filtros convolucionais 1D ou neurônios densos.

Mantendo o **Índice Soma-Produto ($SP$)** padrão do ATLAS Ringer avaliado no joelho (*knee*) da curva ROC (Equação da Seção 1.1):

$$SP_{\text{val}}(\mathbf{x}) = \max_{\tau} \sqrt{\sqrt{PD(\tau) \cdot (1 - FA(\tau))} \cdot \left[ \frac{PD(\tau) + (1 - FA(\tau))}{2} \right]}$$

definimos a função objetivo primal a ser minimizada como o complementar do Índice-$SP$:

$$f(\mathbf{x}) = \mathcal{L}_{\text{SP}}(\mathbf{x}) = 1 - SP_{\text{val}}(\mathbf{x})$$

### 4.2 Vetor de Restrições de Desigualdade de Hardware ($\mathbf{g}(\mathbf{x}) \le \mathbf{0}$)

Para o conjunto de recursos $\mathcal{R} = \{\text{DSP}, \text{LUT}, \text{FF}, \text{BRAM\_bits}\}$, cada restrição de orçamento $B_j$ é expressa na forma canônica $g_j(\mathbf{x}) \le 0$ (ou na forma normalizada adimensional $\tilde{g}_j(\mathbf{x}) = \frac{R_j(\mathbf{x}) - B_j}{B_j} \le 0$ para estabilidade numérica dos multiplicadores):

1. **Restrição de DSPs**:
   $$g_{\text{DSP}}(\mathbf{x}) = \sum_{l=1}^L \text{DSP}_l(w_l, a_l, n_l) - B_{\text{DSP}} \le 0$$
2. **Restrição de LUTs**:
   $$g_{\text{LUT}}(\mathbf{x}) = \sum_{l=1}^L \text{LUT}_l(w_l, a_l, n_l) - B_{\text{LUT}} \le 0$$
3. **Restrição de Flip-Flops (FFs)**:
   $$g_{\text{FF}}(\mathbf{x}) = \sum_{l=1}^L \text{FF}_l(w_l, a_l, n_l) - B_{\text{FF}} \le 0$$
4. **Restrição de Memória On-Chip (BRAM/URAM)**:
   $$g_{\text{Mem}}(\mathbf{x}) = \sum_{l=1}^L \text{Bits}_l(w_l, n_l) - B_{\text{Mem}} \le 0$$

### 4.3 Função Lagrangeana Multi-Recurso e Condições de KKT

A função Lagrangeana associada ao problema restrito na FPGA VU13P, com vetor de multiplicadores duais (preços de sombra) $\boldsymbol{\mu} = [\mu_{\text{DSP}}, \mu_{\text{LUT}}, \mu_{\text{FF}}, \mu_{\text{Mem}}]^T \ge \mathbf{0}$, é dada por:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}) = \left( 1 - SP_{\text{val}}(\mathbf{x}) \right) + \sum_{j \in \mathcal{R}} \mu_j \, g_j(\mathbf{x})$$

No ponto sela ótimo $(\mathbf{x}^*, \boldsymbol{\mu}^*)$, as quatro **Condições de Karush-Kuhn-Tucker (KKT)** exigem:
1. **Estacionaridade (Minimização Primal)**:
   $$\mathbf{x}^* \in \arg\min_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\mu}^*) \quad \left( \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*) = \mathbf{0} \text{ no relaxamento contínuo} \right)$$
2. **Viabilidade Primal**:
   $$g_j(\mathbf{x}^*) \le 0, \quad \forall j \in \{\text{DSP}, \text{LUT}, \text{FF}, \text{Mem}\}$$
3. **Viabilidade Dual**:
   $$\mu_j^* \ge 0, \quad \forall j \in \{\text{DSP}, \text{LUT}, \text{FF}, \text{Mem}\}$$
4. **Folga Complementar (*Complementary Slackness*)**:
   $$\mu_j^* \cdot g_j(\mathbf{x}^*) = 0, \quad \forall j \in \{\text{DSP}, \text{LUT}, \text{FF}, \text{Mem}\}$$

### 4.4 Algoritmo Iterativo Primal-Dual (Ascensão de Subgradiente Dual) e Critério de Parada

Para resolver o problema sobre o espaço discreto de larguras de bits $(w_l, a_l)$ e hiperparâmetros $n_l$:
- **Passo Primal ($t$)**: Para $\boldsymbol{\mu}^{(t)}$ fixo, busca-se a configuração $\mathbf{x}^{(t)}$ que minimiza o custo Lagrangeano penalizado:
  $$\mathcal{L}_{\text{pen}}(\mathbf{x}, \boldsymbol{\mu}^{(t)}) = (1 - SP_{\text{val}}(\mathbf{x})) + \sum_{j \in \mathcal{R}} \mu_j^{(t)} \max\left(0, \tilde{g}_j(\mathbf{x})\right)$$
  (integrando também a alocação analítica de bits por sensibilidade Hessiana $\Omega(\mathbf{w}) = \sum_l c_l 4^{-w_l}$ ponderada pelos custos ativos $\sum_j \mu_j^{(t)} \frac{\partial g_j}{\partial w_l}$).
- **Passo Dual ($t \to t+1$)**: Atualizam-se os multiplicadores de Lagrange via ascensão de subgradiente projetada no ortante não-negativo:
  $$\mu_j^{(t+1)} = \max\left( 0, \, \mu_j^{(t)} + \eta_{\mu} \, \tilde{g}_j(\mathbf{x}^{(t)}) \right), \quad \forall j \in \mathcal{R}$$
- **Critério de Parada KKT**: O loop primal-dual encerra quando simultaneamente:
  1. **Convergência do Índice-SP**: $\left| SP^{(t)} - \frac{1}{m}\sum_{k=1}^m SP^{(t-k)} \right| < \epsilon_{\text{SP}}$;
  2. **Viabilidade Primal Estrita**: $\tilde{g}_j(\mathbf{x}^{(t)}) \le 0, \, \forall j \in \mathcal{R}$;
  3. **Folga Complementar KKT**: $\left| \mu_j^{(t)} \cdot \tilde{g}_j(\mathbf{x}^{(t)}) \right| < \epsilon_{\text{KKT}}, \, \forall j \in \mathcal{R}$.

---

## 5. Ciclo Fechado de Otimização em 3 Etapas (QKeras $\to$ SP/VU13P $\to$ Lagrange/KKT)

O pipeline operacional unifica a teoria das Seções 1 a 4 em um ciclo fechado de três estágios iterativos:

1. **Estágio 1 — Rede Neural Quantizada (QKeras)**:
   - Configura as larguras de bits de pesos ($w_l$), ativações ($a_l$) e unidades ($n_l$) nas camadas `QConv1D` e `QDense`, aplicando transferência de pesos (*Warm-Start* QAT) a partir da referência em ponto flutuante e ajuste fino com decaimento adaptativo de taxa de aprendizado (`ReduceLROnPlateau`).
2. **Estágio 2 — Avaliação de Desempenho Físico e Custo em Silício**:
   - Avalia o **Índice-$SP$** no joelho da curva ROC de validação para computar a perda física $\mathcal{L}_{\text{SP}}(\mathbf{x}^{(t)}) = 1 - SP_{\text{val}}(\mathbf{x}^{(t)})$ e estima analiticamente o consumo de `DSP`, `LUT`, `FF` e `BRAM_bits` na FPGA **Virtex UltraScale+ VU13P**.
3. **Estágio 3 — Núcleo de Otimização Lagrange / KKT**:
   - Compara os recursos consumidos com os tetos de hardware $B_j$ ($g_j(\mathbf{x}^{(t)}) \le 0$), atualiza os **Preços de Sombra** ($\mu_j^{(t+1)}$) via subgradiente dual com projeção de folga complementar, verifica os critérios de parada KKT e calcula analiticamente a próxima alocação de bits $(w_l^{(t+1)}, a_l^{(t+1)})$ via condição de estacionaridade (*water-filling*).
