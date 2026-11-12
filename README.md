# ESN-WiSARD: Reservoir Computing Meets Weightless Neural Networks

Code and results for the paper

> L. Russo, M. De Gregorio, A. Sorgente, C. Gallicchio.
> *Reservoir Computing Meets Weightless Neural Networks.* Submitted to ESANN 2027.

ESN-WiSARD replaces the linear readout of an Echo State Network (ESN), trained by ridge regression, with a weightless
WiSARD readout: the pooled reservoir states are scaled to [0, 1], thermometer-encoded, split by a fixed random mapping
into n-bit tuples that address RAM nodes, and every class has a discriminator whose integer counters are the only
trained quantities. Training is a single pass of integer increments and inference uses table lookups only, without
multiplications.

The repository contains the full pipeline of the paper: model selection of reservoir and readout for every readout,
the ridge readout on quantized states, the diagnostics on correlation and memorization, the energy estimate, and the
scripts that produce Table 2 and Figure 2. The results of all the runs are included, so that the table and the figure
can be regenerated without running any experiment.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_ucr.py all          # downloads the 13 UCR benchmarks into data/ts
```

`reservoirpy` is pinned to 0.4.2 because the reservoir initialisation must match the one used in the paper.
A different folder for the `.ts` files can be set with the environment variable `ESNW_DATA`.

## Quick check

```bash
python model_selection.py smoke      # about 1 minute: 2 reservoir configurations on ItalyPowerDemand
python quantized_ridge.py smoke
```

## Regenerating Table 2 and Figure 2 from the included results

```bash
mkdir -p figures
python aggregate.py                                                    # statistics (Friedman, Nemenyi)
python make_table_results.py results/main/results.csv figures/tab_results.tex
python make_figure_analysis.py results/main/results.csv results/quantized_ridge/results.csv figures/fig_analysis.pdf
python summarize_quantized_ridge.py results/quantized_ridge results/main/results.csv results/quantized_ridge
python energy_estimate.py results/main/results.csv results/energy
```

## Reproducing all the experiments

```bash
bash run_all.sh [n_workers]
```

Every stage runs its tasks in parallel with a process pool and is resumable (completed tasks are skipped when the
same command is launched again). On a 128-core server, the model selection of the main experiment takes a few
hours, the ridge readout on quantized states about 45 minutes, the diagnostics about 20 minutes.

| Script | What it does | Output |
|---|---|---|
| `model_selection.py` | Main experiment: for every benchmark, reservoir size ($N_h \in \{128, 1024\}$), pooling (last or mean state) and readout (Ridge, W-MM, W-QU, W-QG), joint selection of reservoir and readout hyper-parameters by 5-fold cross-validation on the training set (3 random reservoirs), then test on 5 new seeds; control Ridge_bin (ridge on the thermometer code of W-QU); ablations on the tuple size (`e3`) and on hashed RAM nodes (`e4`) | `results/main/` |
| `aggregate.py` | Collects the test records and prints mean accuracies, average ranks, Friedman and Nemenyi tests | `results/main/results.csv` |
| `quantized_ridge.py` | Ridge readout on states quantized with 3 to 6 bits (uniform or equal-frequency quantizer), same protocol | `results/quantized_ridge/` |
| `summarize_quantized_ridge.py` | Comparison with the WiSARD readouts that use the same thresholds (Wilcoxon signed-rank tests) | `results/quantized_ridge/summary.csv` |
| `diagnostics.py` | Principal components of the pooled states, ridge readout on the top components, training and test accuracy of W-QU and of the ridge readout | `results/diagnostics/diagnostics.jsonl` |
| `energy_estimate.py` | Analytical energy and memory estimate of the readouts (45-nm costs from M. Horowitz, ISSCC 2014) | `results/energy/` |
| `make_table_results.py`, `make_figure_analysis.py` | Table 2 and Figure 2 of the paper | `figures/` |
| `esnwisard.py`, `esnw_fast.py` | Data loading, reservoir, scalings, thermometer encoding, WiSARD readout (reference and vectorised implementations) | |
| `fetch_ucr.py` | Download of the UCR benchmarks | `data/` |

### Results files

- `results/main/results.csv`: one row per test run (benchmark, `units` = $N_h$, `seed`, `pooling`, `readout`, `kind`,
  selected hyper-parameters, `acc`, `f1`, memory of the readout). `kind` is `main` for the four readouts of the paper,
  `ridge_bin` for the control, `e3` for the tuple-size ablation and `e4` for the hashed RAM nodes.
- `results/main/best_<benchmark>_N<units>.json`: configurations selected by cross-validation, with their validation
  accuracy.
- `results/quantized_ridge/results.csv`, `summary.csv`, `best_*.json`: the same for the ridge readout on quantized
  states (`quantizer` = `mm` uniform or `qu` equal-frequency, `bits` = 3..6, inf, or selected by cross-validation).
- `results/diagnostics/diagnostics.jsonl`, `results/energy/`: diagnostics and energy estimates.

The validation scores of every configuration (the `select/` folders) are not included; they are recomputed by the
corresponding scripts.

## Implementation notes

- The reservoir is the leaky ESN of ReservoirPy 0.4.2 (10% recurrent and input connectivity, state reset for every
  sequence); `esnw_fast.pooled` runs it in batches with NumPy and returns the same states as `Reservoir.run`.
- The WiSARD readout follows the counter-based formulation with bleaching of the WiSARD classifier library used in
  the original thesis code: thermometer code with $\lfloor Q\,p \rfloor$ active bits per feature, fixed random mapping of
  the bits into tuples, integer counters, and bleaching threshold raised from 1 until a unique winner emerges.
  `FastWiSARD` and `wisard_predict` give the same predictions as the reference class `WiSARD`.

## Citation

```bibtex
@inproceedings{russo2027reservoir,
  title     = {Reservoir Computing Meets Weightless Neural Networks},
  author    = {Russo, Leonardo and De Gregorio, Massimo and Sorgente, Antonio and Gallicchio, Claudio},
  booktitle = {Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year      = {2027},
  note      = {Submitted}
}
```

## License

MIT, see `LICENSE`.
