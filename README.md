# ESN-WiSARD: Reservoir Computing Meets Weightless Neural Networks

Code for the paper

> L. Russo, M. De Gregorio, A. Sorgente, C. Gallicchio.
> *Reservoir Computing Meets Weightless Neural Networks.* Submitted to ESANN 2027.

ESN-WiSARD replaces the linear readout of an Echo State Network (ESN) with a weightless WiSARD readout: the pooled
reservoir states are scaled to [0, 1], thermometer-encoded and split by a fixed random mapping into n-bit tuples that
address RAM nodes, and every class has a discriminator whose integer counters are the only trained quantities.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_ucr.py all          # downloads the UCR benchmarks of the paper into data/ts
```

## Usage

```bash
python experiment.py smoke             # installation check, about one minute
python experiment.py all all [n_workers]   # model selection and test on all the benchmarks of the paper
```

`experiment.py` describes the protocol (reservoir grid, readouts, model selection and test) in its header.
`esnwisard.py` and `esnw_fast.py` contain the data loading, the reservoir, the encodings and the WiSARD readout
(reference and vectorised implementations).

## License

MIT, see `LICENSE`.
