# Fundamentação Teórica e Formulação Matemática

Otimização da precisão (bits de pesos e ativações) da CNN1D compacta do ATLAS L0 Ringer para a FPGA
AMD Virtex UltraScale+ **VU13P**, com sensibilidade de 2ª ordem, condições KKT, solução discreta exata,
decomposição dual lagrangiana e ciclo fechado com QAT. Os números citados vêm de
`notebooks/01_quantized_cnn1d_optimization.ipynb` (região `et5_eta0`, topologia `v2`).

> **Histórico.** As Seções 3–5 substituem a formulação anterior. Cada correção está marcada com
> **[Correção]** e a justificativa. A validação em hardware que motivou a Seção 4 está no repositório
> `cnn-fpga-research`.

---

## 1. Contexto físico e métrica (ATLAS Ringer)

Os anéis concêntricos de energia de cada camada de amostragem do calorímetro são normalizados por evento:

$$\hat{x}_i = \frac{x_i}{\left|\sum_{j=1}^{M} x_j\right|}, \quad i = 1, \dots, M$$

(`src/metrics.py::norm1`; para os dados `v2` usados todos os anéis são não negativos, então isto coincide
com a norma $L_1$). $M = 100, 50, 25$ nas versões `v0`, `v1`, `v2`.

### 1.1 Índice Soma-Produto (SP)

$$SP(PD, FA) = \sqrt{\sqrt{PD \cdot (1 - FA)} \cdot \frac{PD + (1 - FA)}{2}}, \qquad
PD = \frac{TP}{TP + FN},\quad FA = \frac{FP}{FP + TN}$$

avaliado no limiar que maximiza o SP (joelho da ROC). **[Correção]** No holdout, o limiar é o escolhido
na **validação** (`sp_at_threshold`); maximizar o SP sobre o próprio conjunto de teste é otimista.

---

## 2. MACs e BOPs

$$\text{MAC}_{\text{conv}} = L^{\text{out}} K C^{\text{in}} C^{\text{out}}, \qquad \text{MAC}_{\text{dense}} = n^{\text{in}} n^{\text{out}}$$

$$\text{BOPs} = \sum_l \text{MAC}_l \, w_l \, a_{\text{in},l}$$

**[Correção — mapeamento de ativações]** As multiplicações da camada $l$ usam a precisão do tensor que
**entra** em $l$: a entrada da rede para `conv1d_1`, a ReLU de `conv1d_1` para `dense_1` e a ReLU de
`dense_1` para `output_dense`. O dicionário `act_bits` é indexado pela camada cuja saída é quantizada
(`{"conv1d_1": a_1, "dense_1": a_2}`), logo $a_{\text{in},l}$ = `act_bits` da camada **anterior**
(`src/hardware.py::layer_input_bits`). A implementação anterior usava `act_bits[l]` na própria camada $l$
e 8 bits fixos em `output_dense`; com isso, reduzir $a_2$ parecia economizar nos 768 MACs de `dense_1`,
quando na verdade afeta os 8 MACs de `output_dense`.

Para que essa precisão de entrada exista de fato, o modelo QKeras ganhou um quantizador explícito dos
anéis (`input_bits`, `ufixed<8,0>`), formato que eles precisam ter na entrada do FPGA.

---

## 3. Sensibilidade à quantização

### 3.1 Modelo de perturbação

Em torno do ponto treinado em FP32 ($\nabla \mathcal{L} \approx 0$), quantizar o tensor $T_l$ com $b$ bits
introduz o erro $\delta_l = Q_b(T_l) - T_l$ e

$$\Delta \mathcal{L} \approx \tfrac12 \delta^{\top}\mathbf{H}\,\delta \approx \sum_l \tfrac12 \sum_{k \in l} H_{kk}\,\delta_k^2
\approx \sum_l \underbrace{\tfrac12\,\frac{\operatorname{Tr}(\mathbf{H}_{ll})}{n_l}\,\big\|Q_b(T_l) - T_l\big\|_2^2}_{\Omega_l(b)}$$

(termos cruzados entre erros de arredondamento independentes têm média nula; o traço **médio** multiplica o
erro **total**, como no HAWQ-V2/V3 e em Campos et al.). $\operatorname{Tr}(\mathbf{H}_{ll})$ é o traço do
bloco diagonal da camada (soma sobre suas $n_l$ entradas).

### 3.2 O quantizador realmente treinado

`src/models.py` concentra a convenção única usada por treino, custo e sensibilidade:

| tensor | quantizador QKeras | passo $\Delta$ | faixa |
|---|---|---|---|
| pesos/bias | `quantized_bits(b, I, alpha=1)`, $I=\min(2,b-1)$ | $2^{I+1-b}$ | $[-2^I,\ 2^I-\Delta]$ |
| ReLU | `quantized_relu(b, I)` | $2^{I-b}$ | $[0,\ 2^I)$ |
| entrada | `quantized_bits(b, 0, keep_negative=False)` | $2^{-b}$ | $[0, 1)$ |

$\|Q_b(T)-T\|^2$ é calculado **com esses objetos** (não com uma fórmula de ruído), então saturação, a faixa
fixa e a grade grossa de 2–3 bits entram exatamente.

### 3.3 Forma contínua $\Omega_l \approx c_l 4^{-b}$

Com ruído uniforme $\mathbb{E}\|\delta\|^2 = n_l\Delta^2/12$ e $\Delta = \kappa\,2^{-b}$:

$$\Omega_l(b) = \tfrac12\frac{\operatorname{Tr}(\mathbf{H}_{ll})}{n_l}\,\frac{n_l\kappa^2 4^{-b}}{12}
\;\Longrightarrow\; c_l = \frac{\operatorname{Tr}(\mathbf{H}_{ll})\,\kappa^2}{24},
\qquad \kappa = 2^{I+1} = 8 \ (b \ge 3)$$

**[Correção — fator $n_l$]** A versão anterior usava $c_l = \operatorname{Tr}(\mathbf{H}_l)\,n_l r_l^2/3$ com o
traço **total** do Hutchinson, contando $n_l$ duas vezes. Isso favorece camadas grandes por um fator ~$n_l$
(768 pesos em `dense_1` vs 24 em `conv1d_1`). Com as funções originais, a alocação `[2,5,8]` reaparece
em 5/5 repetições; sem o fator $n_l$ sai `[4,4,8]`. O "ótimo KKT" anterior era efeito desse erro.

**[Correção — faixa]** $r_l = \max|w|$ supunha um quantizador adaptativo $[-r_l, r_l]$, mas o QKeras com
`alpha=1` usa faixa fixa $2^I$; com ela, $c_l$ depende só do traço (`range_mode="qkeras"`, padrão).

### 3.4 Hutchinson bloco a bloco

$$\operatorname{Tr}(\mathbf{H}_{ll}) \approx \frac{1}{M}\sum_{j=1}^{M} \mathbf{z}_j^{\top}\mathbf{H}_{ll}\mathbf{z}_j,
\qquad \mathbf{z}_j \in \{-1,+1\}^{n_l}$$

**[Correção]** Cada camada é sondada separadamente (um HVP por camada e sorteio, $M=32$, 4096 eventos).
O vetor conjunto anterior somava termos cruzados $\mathbf{z}_l^\top\mathbf{H}_{lm}\mathbf{z}_m$, de média
nula mas variância alta, o que tornava negativo o traço estimado da camada de saída. Ele era então
truncado em $10^{-8}$.

### 3.5 Ativações

A mesma fórmula vale para a ativação $A_m$ (lote × atributos), com o Hessiano da perda em relação a $A_m$
(`compute_activation_hessian_traces`). Como a perda é a média do lote, $\mathbf{H}$ é bloco-diagonal por
evento e o traço soma as curvaturas de todas as entradas.

### 3.6 Validação do proxy (resultado)

Comparando $\Omega_l(b)$ com o aumento de perda **medido** ao quantizar só aquele tensor (PTQ, validação):

* correlação de postos (Spearman, $b \le 8$): 0,75 no total, 0,52 em $4 \le b \le 8$;
* o modelo de Taylor quebra em poucos bits: em `dense_1` com 2–3 bits, $\Omega$ subestima ~70×
  (0,075 vs 5,3);
* em `a:dense_1`, $\Omega$ tem um patamar ~$10^{-2}$ causado pela saturação da ReLU em $2^I = 4$, que
  quase não afeta a perda (superestima até ~50×);
* acima de ~8 bits o $\Delta L$ medido fica abaixo da resolução da avaliação (~$10^{-5}$).

Por isso o problema discreto aceita duas tabelas de objetivo: `hessian` ($\Omega$ de Taylor) e `measured`
($\Delta L$ medido, truncado em 0 e tornado não crescente em $b$ por `monotone_table`).

---

## 4. Modelo de custo na VU13P (hls4ml `io_parallel`)

### 4.1 Por que o modelo anterior foi substituído

A rede foi sintetizada com hls4ml 1.3 → Vitis HLS → Vivado 2026.1.1 (place & route out-of-context,
240 MHz, `ReuseFactor=1`, Conv1D totalmente paralela). As regras anteriores
($\text{LUT} = \text{MAC}\,w a/18$, $\text{DSP} = \lceil \text{MAC}/64 \rceil$, $\text{BRAM} = P\,w$):

| config | DSP est. → real | LUT est. → real | BRAM est. → real |
|---|---|---|---|
| u16 | 23 → 1.241 | 1.221 → 29.571 | 13.072 bits → 1 RAMB18 |
| u8 | 0 → 0 | 4.892 → 23.309 | 6.536 bits → 1 RAMB18 |
| `[2,5,8]` | 0 → 0 | 2.268 → 6.923 | 4.016 bits → 1 RAMB18 |

Em `io_parallel`/`Latency` os pesos são **constantes** na lógica. Consequências:

* o custo dominante são as árvores de soma de cada neurônio e a lógica que consome os acumuladores
  (ReLU + pooling), não um multiplicador genérico $w \times a$;
* peso nulo não custa nada e potência de 2 é deslocamento (fio);
* multiplicador constante com $d$ dígitos CSD não nulos custa $d-1$ somadores, ou 1 DSP48E2 se for largo
  (> ~10 bits) e não trivial;
* não há pesos em BRAM: a única RAM é a tabela da sigmoide.

### 4.2 Modelo estrutural

Para a camada $l$ com $P_l$ posições em paralelo, fan-in $n_l$, $n^{\text{out}}_l$ neurônios e códigos inteiros
$k$ dos pesos:

$$\text{somadores}_l = P_l \sum_{o}\Big[(\text{termos}_o - 1) + \sum_{k \in o,\ \text{LUT}} (\text{csd}(k) - 1)\Big],
\qquad \text{largura}_l = a_{\text{in},l} + w_l + \tfrac12\lceil \log_2 n_l \rceil$$

$$\text{ReLU}_l = P_l\, n^{\text{out}}_l \,\big(a_{\text{in},l} + w_l + \lceil \log_2 n_l \rceil\big) \quad
\text{(bits de acumulador que entram na ReLU/pooling; 0 se não houver ReLU)}$$

$$\text{DSP} = \sum_l P_l\,\#\{k : \max(w_l, a_{\text{in},l}) > 10,\ \text{csd}(k) \ge 3\}$$

$$\text{LUT} = \beta_{\text{LUT}} \sum_l \text{somadores}_l\,\text{largura}_l + \gamma_{\text{LUT}} \sum_l \text{ReLU}_l, \qquad
\text{FF} = \beta_{\text{FF}} \sum_l \text{somadores}_l\,\text{largura}_l + \delta_{\text{FF}}\,\text{DSP} + \gamma_{\text{FF}},
\qquad \text{RAMB18} = 1$$

### 4.3 Calibração, validação fora da amostra e domínio

Pontos de calibração (`src/calibration/vu13p_hls4ml_synthesis.json`, gerado por
`cnn-fpga-research/scripts/export_calibration.py`): pesos inteiros efetivamente sintetizados + utilização
pós-rota de u16, u8, `[2,5,8]`, `mp_v1` = `[3,4,16]/[8,12]` e `mp_b10k` = `[4,6,16]/[8,12]` (mais
`[2,5,8]` com PF = 1, fora do domínio). Mínimos quadrados: $\beta_{\text{LUT}} = 0{,}395$ LUT/bit de somador,
$\gamma_{\text{LUT}} = 1{,}33$ LUT/bit de ReLU, $\beta_{\text{FF}} = 0{,}171$, $\delta_{\text{FF}} = 6{,}4$ FF/DSP,
$\gamma_{\text{FF}} = 956$.

| ponto | LUT ajuste (LOO) | FF ajuste (LOO) | DSP |
|---|---|---|---|
| u16 | −0,4 % (−0,9 %) | 0,0 % (+26,7 %) | +0,3 % |
| u8 | +0,4 % (+2,2 %) | −1,1 % (−27,7 %) | 0 |
| `[2,5,8]` | +20,8 % (+24,6 %) | +4,0 % (+6,2 %) | 0 |
| mp_v1 | −6,0 % (−8,8 %) | −12,4 % (−21,8 %) | 8 vs 7 |
| mp_b10k | −5,5 % (−6,2 %) | +8,9 % (+12,1 %) | 8 vs 10 |
| `[2,5,8]` PF = 1 (fora do domínio) | −68 % | −82 % | 0 |

**Validação com previsões congeladas antes da síntese** (`cnn-fpga-research/scripts/check_cost_model.py`):

* **Rodada 1** (modelo só com somadores, ajustado em u16, u8, `[2,5,8]`): mp_v1 5.072 → 7.785 LUT (−35 %),
  mp_b10k 10.251 → 12.011 (−15 %). **[Correção]** O modelo era otimista em redes pequenas: a lógica de
  ReLU + pooling sobre as 200 saídas da conv (~3–5 mil LUT) não encolhe com as árvores de soma. Daí o termo
  $\text{ReLU}_l$; entre as formas testadas, foi a de menor erro LOO (RMS 12 % em LUT contra 19–21 %).
* **Rodada 2** (modelo atual, 5 pontos): a configuração final `[4,4,16]/[8,16]`, mantida fora do ajuste,
  foi prevista em 7.893 LUT / 8 DSP / 2.728 FF e sintetizou com **6.783 LUT (+16 %, conservador) / 7 DSP /
  2.585 FF (+6 %)**.

**Margem de segurança.** A pior subestimação LOO de cada recurso vira margem: $m_{\text{LUT}} = 8{,}8\,\%$,
$m_{\text{FF}} = 27{,}7\,\%$, $m_{\text{DSP}} = 20\,\%$ (`CostModel.margins`). As restrições usam
$B_j^{\text{ef}} = B_j/(1+m_j)$ (restrição robusta à incerteza do modelo).

**Domínio:** hls4ml 1.3 `io_parallel`, `Latency`, `ReuseFactor=1`, Conv1D com PF = largura de saída,
VU13P a 240 MHz, esta topologia. Fora dele (PF/RF menores, outra rede) é preciso sintetizar novos pontos.

**Previsão antes do treino.** O otimizador estima o custo quantizando os pesos **FP32** (PTQ). Erro contra o
custo com os pesos pós-QAT: mediana −3 %, pior caso 13 %. O ciclo fechado recalcula o custo com os pesos
treinados.

### 4.4 Orçamento por instância

$$B_j = f_{\text{disp}} \cdot \frac{B_{j,\text{chip}}}{N_{\text{inst}}}, \qquad j \in \{\text{DSP}, \text{LUT}, \text{FF}\}$$

$N_{\text{inst}}$ cópias do classificador dividem a fração $f_{\text{disp}}$ da VU13P. No notebook
$N_{\text{inst}} = 100$, $f_{\text{disp}} = 0{,}6$ ($B_{\text{LUT}} = 10.368$, $B_{\text{FF}} = 20.736$,
$B_{\text{DSP}} = 73{,}7$; efetivos 9.529 / 16.243 / 61,4) é **hipótese ilustrativa**; a fronteira da Seção
5.6 mostra a solução para vários orçamentos. **[Correção]** O teto anterior de 2.500 LUT foi escolhido para
que 8 bits violasse e a solução proposta passasse; pela síntese, essa solução usa 6.923 LUT e violaria o
próprio teto. A BRAM sai das restrições (constante = 1 RAMB18). Latência (9–12 ciclos, 37–50 ns) e II = 1
não são restritivas no domínio calibrado.

---

## 5. Problema de otimização e algoritmos

### 5.1 Formulação

$$\min_{\mathbf{x}} \ \Omega(\mathbf{x}) = \sum_l \Omega^w_l(w_l) + \sum_m \Omega^a_m(a_m)
\quad \text{s.a.} \quad g_j(\mathbf{x}) = \frac{R_j(\mathbf{x}) - B^{\text{ef}}_j}{B^{\text{ef}}_j} \le 0,\ j \in \{\text{DSP}, \text{LUT}, \text{FF}\}$$

com $w_l, a_m \in \{2,\dots,8,10,12,16\}$ (3 pesos + 2 ativações: $10^5$ combinações). $\Omega$ é um proxy
de $1 - SP$ calculado antes do treino; o SP entra no ciclo fechado (5.6).

### 5.2 Separabilidade

O custo da camada $l$ depende de $(w_l, a_{\text{in},l})$, e $a_{\text{in},l}$ é a ReLU da camada anterior.
Agrupando cada camada com a ativação que a alimenta,
$y_1 = w_{\text{conv}}$, $y_2 = (w_{\text{dense}}, a_{\text{conv}})$, $y_3 = (w_{\text{out}}, a_{\text{dense}})$,
tanto $\Omega$ quanto $R_j$ ficam somas de termos de um grupo só (`build_choice_groups`).

### 5.3 Relaxação contínua (water-filling KKT)

Subproblema só de pesos, ativações fixas em 8 bits, custo de LUT linearizado por camada
($R_l(w) \approx p_l + q_l w$, `linear_cost_coefficients`) e $\Omega_l = c_l 4^{-w_l}$:

$$\min_{\mathbf{b}} \sum_l c_l 4^{-b_l} \quad \text{s.a.} \quad \sum_l q_l b_l \le B' ,\quad b_{\min} \le b_l \le b_{\max}$$

O objetivo é estritamente convexo ($c_l(\ln 4)^2 4^{-b} > 0$), as restrições são afins e vale Slater; logo
KKT é necessário e suficiente. Estacionaridade:
$-c_l \ln 4 \cdot 4^{-b_l} + \lambda q_l - \nu_l^- + \nu_l^+ = 0$, de onde

$$b_l^*(\lambda) = \operatorname{clip}\!\Big(\tfrac12\log_2\frac{c_l \ln 4}{\lambda q_l},\ b_{\min},\ b_{\max}\Big).$$

$h(t) = \sum_l q_l b_l^*(2^t) - B'$ com $t = \log_2\lambda$ é não crescente e **linear por partes**, com
inclinação $-\tfrac12\sum_{l\ \text{interior}} q_l$. O método de Newton em $t$ é exato em cada trecho; ele é
salvaguardado por um intervalo de bisseção.

**Resultado.** Bits contínuos $(3{,}84;\ 4{,}37;\ 3{,}59)$, $\lambda^* = 1{,}3\times10^{-5}$, resíduos de
estacionaridade ~$10^{-18}$. Newton converge em **4** iterações e a bisseção em 31. Arredondado (piso +
preenchimento guloso): $(4,4,8)$, $\Omega = 0{,}040$, 7.298 LUT. O ótimo discreto do mesmo subproblema é
$(4,5,4)$, com $\Omega = 0{,}022$ e 9.522 LUT: arredondar sobre o custo linearizado perde 83 % em $\Omega$.

### 5.4 Solução discreta exata

Com $10^5$ combinações, a enumeração vetorizada (`solve_exact`) leva ~1 ms e dá o ótimo global, que é a
verdade de referência (equivale ao ILP do HAWQ-V3 para este tamanho).

### 5.5 Relaxação lagrangiana (primal-dual com preços-sombra)

$$L(\mathbf{x}, \boldsymbol\mu) = \Omega(\mathbf{x}) + \sum_j \mu_j g_j(\mathbf{x}), \qquad
q(\boldsymbol\mu) = \min_{\mathbf{x}} L(\mathbf{x}, \boldsymbol\mu) \le \Omega^* \quad (\boldsymbol\mu \ge 0)$$

* **Passo primal (exato):** $\mathbf{x}(\boldsymbol\mu)$ separa em três $\arg\min$ independentes por grupo.
* **Passo dual:** $q$ é côncava e $\mathbf{g}(\mathbf{x}(\boldsymbol\mu))$ é um supergradiente:
  $\mu_j \leftarrow \max(0, \mu_j + \eta_t g_j)$, com passo de Polyak
  $\eta_t = \theta_t(\text{UB} - q_t)/\|\mathbf{g}\|^2$ e $\theta_t$ reduzido pela metade quando o limite dual
  estagna (Held-Karp).
* **Limite superior:** cada $\mathbf{x}(\boldsymbol\mu)$ passa por reparo guloso de viabilidade (redução
  estrita da violação total) e por melhoria local (heurística lagrangiana).
* **Certificado KKT (teorema de Everett):** se $\mathbf{x}(\boldsymbol\mu)$ é viável, $\boldsymbol\mu \ge 0$
  e $\mu_j g_j = 0\ \forall j$, então $\Omega(\mathbf{x}) = q(\boldsymbol\mu)$ e $\mathbf{x}$ é ótimo global, também
  para variáveis inteiras. Se esse par não existe há **gap de dualidade** e só se têm limites
  $\text{LB} \le \Omega^* \le \text{UB}$.

**[Correção]** O laço anterior não minimizava o Lagrangiano. O passo primal usava o orçamento
$B_{\text{LUT}}\cdot 18$, constantes fixas 0,35 e 0,58 e a regra `dense_1 = 6 bits`, e o SP nunca entrava.
O multiplicador decaía de propósito quando havia folga, o que cumpria a folga complementar
artificialmente: a "convergência KKT em 5 iterações" era só esse decaimento.

**Resultado** ($B^{\text{ef}}$ da Seção 4.4):

| objetivo | ótimo exato | dual + heurística | LB (dual) | gap | $\mu$ |
|---|---|---|---|---|---|
| hessian | $w=[4,5,4]$, $a=[8,8]$, $\Omega=0{,}0218$, 9.522 LUT | igual | 0,0215 | 1,3 % | $\mu_{\text{LUT}}=0{,}028$, $\mu_{\text{DSP}}=\mu_{\text{FF}}=0$ |
| measured | $w=[4,5,12]$, $a=[8,5]$, $\Omega=0{,}0097$, 9.522 LUT | igual | 0,0097 | 0,3 % | $\mu_{\text{LUT}}=0{,}020$, $\mu_{\text{DSP}}=\mu_{\text{FF}}=0$ |

Os preços-sombra têm a leitura esperada: só LUT é o recurso limitante; FF e DSP têm folga e $\mu = 0$
(folga complementar). O gap é inerente ao problema inteiro, e a enumeração confirma que o primal encontrado
é o ótimo. Neste problema pequeno a heurística de reparo + melhoria já chega ao ótimo na 1ª iteração; o valor
da fase dual está no limite inferior e nos preços-sombra.

### 5.6 Ciclo fechado com QAT

$\Omega$ descreve a degradação **antes** do QAT, e o QAT recupera boa parte dela: `[2,5,8]` tem
$\Omega_{\text{medido}} = 0{,}55$, mas chega a SP 97,65 % após o QAT. Por isso o laço:

1. **Fronteira:** resolve o problema exato para $B_{\text{LUT}} \in \{4, 6, 8, 10{,}4, 14, 20\}$ mil (com a
   margem), nos dois objetivos.
2. **Triagem:** treina com QAT (warm-start do FP32) cada solução distinta, as uniformes u3–u8 e a `[2,5,8]`
   antiga; mede o SP de validação e recalcula o custo com os pesos treinados.
3. **Lista curta:** até 4 candidatos viáveis a ≤ 0,2 p.p. do melhor.
4. **Decisão por CV pareada** (Seção 5.7): a configuração de menor LUT médio cuja diferença pareada em relação
   ao melhor candidato é ≥ −0,1 p.p.

**[Correção]** Escolher entre quase-empates numa única partição é ruído. Com a regra antiga (menor custo a
≤ 0,1 p.p. do melhor na validação) saiu `[3,3,2]/[7,10]`, que na CV perde 0,38 ± 0,21 p.p. para o u8
(~4 erros-padrão).

Resultados da triagem (validação):

* a precisão mista fica em 97,7–97,8 %; a uniforme cai rápido: u5 96,18 %, u4 93,91 %, u3 84,95 %;
* soluções do ótimo exato no limite do orçamento ficam frágeis: `[4,5,4]/[8,8]` foi prevista em 9.522 LUT
  e, com os pesos pós-QAT, custa 9.730 > $B^{\text{ef}}_{\text{LUT}}$;
* correlação de postos entre $\Omega$ e a perda de SP pós-QAT: 0,67 (hessian) e 0,65 (measured).

| lista curta (w / a) | SP CV | vs u8 (pareado) | LUT médio |
|---|---|---|---|
| `[3,3,2] / [7,10]` | 97,34 % | −0,38 p.p. | 5,7 mil |
| `[2,5,8] / [8,8]` (antigo) | 97,48 % | −0,24 p.p. | 8,6 mil |
| **`[4,4,16] / [8,16]`** (final) | **97,65 %** | **−0,07 ± 0,09 p.p.** | **8,1 mil** |
| `[4,4,16] / [8,12]` | 97,66 % | −0,06 p.p. | 8,2 mil |

### 5.7 Protocolo estatístico

**[Correção]** Na validação cruzada, o FP32 de warm-start de cada fold é treinado só com o treino do fold.
Antes, um único FP32 treinado em 80 % de `X_dev` alimentava todos os folds, cujas validações estavam nesses
80 %. Os candidatos da lista curta, o FP32 e o u8 são treinados nos mesmos folds (diferenças pareadas).

**Resultado final** (`w = [4,4,16]`, `a = [8,16]`, entrada 8 b):

* CV: SP = 97,65 ± 0,24 %; diferença pareada final − u8 = −0,07 ± 0,09 p.p. (erro padrão 0,04) e
  final − FP32 = +0,04 ± 0,07 p.p.;
* holdout com limiar da validação: 97,62 % (FP32: 97,81 %);
* custo previsto 7,6–8,6 mil LUT conforme o fold; **sintetizado** (pesos do melhor fold): 6.783 LUT,
  7 DSP, 2.585 FF, timing fechado a 240 MHz (folga 0,95 ns), latência 9 ciclos (37,5 ns), II = 1.
  O u8 usa 23.309 LUT.

---

## 6. Limitações e próximos passos

* **Calibração:** 5 pontos no domínio, mais 1 fora, e um teste fora da amostra por rodada. Sintetizar outros
  PF/`ReuseFactor` permite incluir o paralelismo como variável de decisão e a latência/II como restrição.
* **Aditividade:** $\Omega$ ignora interações entre camadas, e sua correlação com o SP pós-QAT é moderada
  (0,65–0,67). O proxy serve para gerar candidatos; a decisão fica com a CV.
* **Custo dependente do treino:** varia ~±10 % entre retreinos (zeros/CSD); por isso a margem e a
  verificação de viabilidade em todos os folds.
* **Orçamento:** $B_j$ precisa vir do plano do firmware L0 (instâncias por FPGA, recursos de outros blocos).
* **Arquitetura:** o modelo supõe pesos constantes por instância. Um caminho de dados compartilhado entre os
  25 especialistas (pesos em memória) tem outra estrutura de custo e exige nova calibração.
