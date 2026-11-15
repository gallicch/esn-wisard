# ESN-WiSARD: Reservoir Computing Meets Weightless Neural Networks

Code for the paper

> L. Russo, M. De Gregorio, A. Sorgente, C. Gallicchio.
> *Reservoir Computing Meets Weightless Neural Networks.* Submitted to ESANN 2027.

ESN-WiSARD replaces the linear readout of an Echo State Network (ESN) with a weightless WiSARD readout: the pooled
reservoir states are scaled to [0, 1], thermometer-encoded and split by a fixed random mapping into n-bit tuples that
address RAM nodes, and every class has a discriminator whose integer counters are the only trained quantities.

## Benchmarks

The 10 univariate UCR benchmarks of the paper, in which the class depends on the temporal development of the signal
rather than on an aligned shape: ACSF1, FordA, FordB, FreezerRegularTrain, LargeKitchenAppliances,
RefrigerationDevices, SemgHandMovementCh2, SyntheticControl, Trace, TwoPatterns. Every series is z-normalized
(see `fetch_ucr.py`).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_ucr.py all          # downloads the 10 benchmarks into data/ts
```

## Usage

```bash
python experiment.py smoke                  # installation check, about one minute
python experiment.py all all [n_workers]    # model selection and test on the 10 benchmarks
python analyze.py results results/paper     # Table 2, Figure 2 and the numbers of the paper
```

`experiment.py` describes the protocol (reservoir grid, readouts, model selection and test) in its header.
`esnwisard.py` and `esnw_fast.py` contain the data loading, the reservoir, the encodings and the WiSARD readout
(reference and vectorised implementations).

## Results of the paper

`paper_results/` holds the selected configurations (`best_*.json`), the test results over 10 seeds (`results.csv`)
and the output of `analyze.py`: Table 2 (`tab_results.tex`), Figure 2 (`fig_tradeoff.tex`), the numbers quoted in
the text (`summary.json`) and the energy estimate of the readouts (`energy_estimates.csv`). The tables and the figure
can be regenerated from the test results with `python analyze.py paper_results <out_dir>`.

## License

MIT, see `LICENSE`.
