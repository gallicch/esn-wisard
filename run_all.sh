#!/usr/bin/env bash
# Full pipeline of the paper. Usage: bash run_all.sh [n_workers]
# The model selection is the expensive part (a few hours on a 128-core server); every stage is resumable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
NW="${1:-$(nproc)}"
python fetch_ucr.py all                              # 13 UCR benchmarks into data/ts
python model_selection.py all all "$NW"             # main experiment + ablations -> results/main
python aggregate.py                                  # results/main/results.csv, statistics
python quantized_ridge.py all all "$NW"             # ridge on quantized states -> results/quantized_ridge
python summarize_quantized_ridge.py results/quantized_ridge results/main/results.csv results/quantized_ridge
python diagnostics.py all "$NW"                      # correlation and memorization -> results/diagnostics
python energy_estimate.py results/main/results.csv results/energy
mkdir -p figures
python make_table_results.py results/main/results.csv figures/tab_results.tex
python make_figure_analysis.py results/main/results.csv results/quantized_ridge/results.csv figures/fig_analysis.pdf
