# Notas

## Encaminhamentos

CNN1D compacta (`v2`, 25 anéis) para a FPGA **Xilinx Virtex UltraScale+ VU13P**, com alocação de bits por
**Multiplicadores de Lagrange e condições KKT**:

1. **Sensibilidade** $\Omega_l(b)$ de pesos e ativações: Hessiana (Hutchinson) e $\Delta L$ medido;
2. **Custo na VU13P** (`DSP`, `LUT`, `FF`) por modelo estrutural **calibrado com sínteses reais**
   (hls4ml → Vitis HLS → Vivado, repositório `cnn-fpga-research`), com margem de segurança pelo erro LOO;
3. **Otimização restrita**: water-filling KKT (contínuo), enumeração exata (discreto) e decomposição dual
   com preços-sombra $\mu_j$ e certificado KKT;
4. **Ciclo fechado**: QAT dos candidatos da fronteira (triagem), decisão por CV pareada sem vazamento,
   holdout com limiar fixado na validação, síntese de confirmação.

---

## Correções da revisão (branch `fix/activation-mapping`)

| problema | correção |
|---|---|
| $c_l = \mathrm{Tr}(H)\,n_l r^2/3$ contava $n_l$ duas vezes; o "ótimo" `[2,5,8]` vinha disso | $\Omega_l = \tfrac12 (\mathrm{Tr}/n_l)\,\|Q_b - T\|^2$ com o quantizador real; $c_l = \mathrm{Tr}\,\kappa^2/24$ |
| bits de ativação aplicados à camada errada no custo | precisão do tensor **de entrada** de cada camada; quantizador explícito da entrada (8 b) |
| custo LUT/DSP/BRAM errava 3–54× frente à síntese | modelo estrutural (somadores, zeros, CSD, ReLU, DSP) calibrado em 5 sínteses; erro fora da amostra +16 % em LUT |
| passo primal com constantes fixas, sem minimizar o Lagrangiano; decaimento forçava a folga complementar | decomposição dual exata por grupo, passo de Polyak, limites LB/UB e certificado KKT honesto |
| teto de 2.500 LUT escolhido para o resultado desejado (a solução real usa 6.923) | $B_j = f\cdot\text{chip}_j/N$ explícito, orçamento efetivo $B_j/(1+m_j)$, fronteira para vários orçamentos |
| Hutchinson com vetor conjunto (traço negativo na saída) | sondagem bloco a bloco |
| SP não realimentava; vazamento do warm-start na CV; limiar escolhido no holdout | SP pós-QAT na triagem; decisão por CV pareada; FP32 por fold; limiar da validação |
| `.keras` exportado não recarregava (QKeras) | exportador grava também `.h5` (`qkeras.utils.load_qmodel`) |

## Resultados (`notebooks/01_quantized_cnn1d_optimization.ipynb`, `et5_eta0`)

* **Configuração final:** $w = [4, 4, 16]$, $a = [8, 16]$, entrada 8 b.
* **Sintetizada (VU13P, 240 MHz, pós-rota):** 6.783 LUT, 7 DSP, 2.585 FF; timing fechado (folga 0,95 ns);
  latência 9 ciclos (37,5 ns); II = 1. O u8 uniforme usa 23.309 LUT (3,4× mais). O orçamento ilustrativo é
  $B_{\text{LUT}} = 10{,}4$ mil.
* **SP:** CV 97,65 ± 0,24 % e holdout 97,62 %. Diferença pareada final − u8 = −0,07 ± 0,09 p.p. (erro padrão
  0,04) e final − FP32 = +0,04 ± 0,07 p.p.
* **Precisão mista × uniforme (validação):** a mista fica em 97,7–97,8 %; a uniforme cai para 96,2 % (u5),
  93,9 % (u4) e 85,0 % (u3).
* **Modelo de custo:** a versão de 3 pontos subestimou redes pequenas em até 35 % (lógica de ReLU/pooling).
  A atual, com 5 pontos, previu a final em 7.893 LUT, contra 6.783 sintetizados (+16 %, conservador).
* **Dual × exato:** a decomposição dual encontra o ótimo da enumeração nos dois objetivos (gap dual de 1,3 % e
  0,3 %). Só $\mu_{\text{LUT}} > 0$.
* **Water-filling:** Newton em 4 iterações contra 31 da bisseção; o arredondamento perde 83 % em $\Omega$
  frente ao ótimo discreto.
* **Proxy:** a Hessiana falha em ≤ 3 bits (até ~70× abaixo do medido). Nenhum dos dois proxies ordena bem os
  candidatos depois do QAT (Spearman 0,65–0,67), por isso a decisão é por CV.

## Pendências

* Mais pontos de síntese (PF/`ReuseFactor`) para incluir paralelismo e latência no problema.
* Definir $N_{\text{inst}}$ e $f_{\text{disp}}$ com o plano de firmware do L0.
* Decidir entre circuito dedicado por região (pesos constantes, modelo atual) e caminho de dados
  compartilhado pelos 25 especialistas (outro modelo de custo).
* `best_model.onnx` ainda é um JSON de fallback quando `tf2onnx` não está instalado (`src/exporter.py`).
