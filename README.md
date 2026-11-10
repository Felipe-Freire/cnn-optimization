# cnn-optimization

Alocação de precisão (bits de pesos e ativações) da CNN1D compacta do ATLAS L0 Ringer sob restrições de
recursos da FPGA AMD Virtex UltraScale+ VU13P, usando sensibilidade de 2ª ordem (Hessiana), condições KKT,
enumeração exata e decomposição dual lagrangiana, com ciclo fechado de QAT.

* Teoria e deduções: [`docs/theory_and_formulation.md`](docs/theory_and_formulation.md)
* Notas do projeto e resultados: [`Notas.md`](Notas.md)
* Validação em hardware (hls4ml → Vitis HLS → Vivado) e origem dos dados de calibração do modelo de custo:
  repositório `cnn-fpga-research` → `src/calibration/vu13p_hls4ml_synthesis.json`

## Estrutura

```
src/
├── models.py         # CNN1D FP32/QKeras e a convenção única dos quantizadores
├── metrics.py        # anéis v0/v1/v2, norm1, índice SP, SPCallback
├── sensitivity.py    # Hutchinson (pesos e ativações), Ω_l(b), c_l, ΔL medido (PTQ)
├── hardware.py       # modelo de custo calibrado na VU13P (DSP, LUT, FF), orçamento por instância
├── kkt_allocator.py  # water-filling KKT, enumeração exata, decomposição dual, certificado KKT, fronteira
├── training.py       # FP32, QAT com warm-start, avaliação com limiar fixo
├── exporter.py / visualization.py
└── calibration/vu13p_hls4ml_synthesis.json
scripts/generate_notebook.py, scripts/generate_results_notebook.py   # geram os notebooks
notebooks/01_quantized_cnn1d_optimization.ipynb                       # pipeline completo (executado)
notebooks/02_results_and_visualizations.ipynb                         # leitura dos artefatos exportados
```

## Como rodar

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q

# dados: <dir>/mc21_13p6TeV.Zee.JF17.2sigma.5M.et5_eta0.h5 (default: ./data)
export RINGER_DATA_DIR=/caminho/para/os/h5
.venv/bin/python scripts/generate_notebook.py          # regenera o notebook a partir do script
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_quantized_cnn1d_optimization.ipynb
```

Nenhuma ferramenta de FPGA é necessária aqui: o modelo de custo é ajustado a partir das sínteses já
armazenadas em `src/calibration/`. Para estender o domínio do modelo (outros `ReuseFactor`/PF, outros bits),
gere novos pontos no `cnn-fpga-research` (`scripts/export_calibration.py`).
