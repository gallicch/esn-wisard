"""Aggregate the test records of the main experiment and print the statistics reported in the paper.

Reads results/main/test_*_N*_seed*.jsonl (written by model_selection.py test) and writes results/main/results.csv;
if the raw records are not present, it reads the results.csv shipped with the repository.
Prints, for every N_h and pooling: mean accuracy of the four readouts and of Ridge_bin, average ranks,
Friedman p-value and Nemenyi critical difference (exact ties are kept as ties).
Usage: python aggregate.py [results/main]"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "main")
MAIN = ["ridge", "wisard_minmax", "wisard_quantile_uniform", "wisard_quantile_gaussian"]
NAME = {"ridge": "Ridge", "wisard_minmax": "W-MM", "wisard_quantile_uniform": "W-QU", "wisard_quantile_gaussian": "W-QG",
        "ridge_on_binary": "Ridge_bin"}
Q05 = {4: 2.569}   # Nemenyi critical value for 4 methods, alpha = 0.05 (Demsar, 2006)

files = glob.glob(os.path.join(RES, "test_*_N*_seed*.jsonl"))
if files:
    df = pd.DataFrame([json.loads(l) for f in files for l in open(f)]).drop_duplicates(["key", "dataset", "units", "seed"])
    for k in ["id", "sr", "input_scaling", "lr", "bias"]:
        df["cfg_" + k] = df.cfg.apply(lambda c: c[k])
    df = df.drop(columns=["cfg"])
    df.to_csv(os.path.join(RES, "results.csv"), index=False)
else:
    df = pd.read_csv(os.path.join(RES, "results.csv"))
print("records:", len(df), "| benchmarks:", df.dataset.nunique())

d = df[df.kind.isin(["main", "ridge_bin"])]
for u in sorted(d.units.unique()):
    for p in ["last", "mean"]:
        A = d[(d.units == u) & (d.pooling == p)].groupby(["dataset", "readout"]).acc.mean().unstack()
        M = np.round(A[MAIN].values, 6)
        ranks = pd.DataFrame(-M).rank(axis=1, method="average").values.mean(0)
        pv = friedmanchisquare(*M.T).pvalue
        cd = Q05[4] * np.sqrt(4 * 5 / (6 * len(A)))
        print(f"\nN_h = {u}, {p} state: Friedman p = {pv:.4f}, Nemenyi CD = {cd:.2f}")
        print("  mean acc:  " + "  ".join(f"{NAME[r]} {A[r].mean():.2f}" for r in MAIN + ["ridge_on_binary"]))
        print("  avg rank:  " + "  ".join(f"{NAME[r]} {x:.2f}" for r, x in zip(MAIN, ranks)))
        print("  rank diff from Ridge: " + "  ".join(f"{NAME[r]} {ranks[i] - ranks[0]:.2f}" for i, r in enumerate(MAIN) if i))
