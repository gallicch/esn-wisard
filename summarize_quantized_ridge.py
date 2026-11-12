"""Summary of the ridge readout on quantized states (quantized_ridge.py) against the WiSARD readouts of the main experiment.
Usage: python summarize_quantized_ridge.py results/quantized_ridge results/main/results.csv results/quantized_ridge"""
import sys, os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RQ, MAIN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
files = glob.glob(os.path.join(RQ, "test_*.jsonl"))
if files:      # raw test records written by quantized_ridge.py
    rq = pd.DataFrame([json.loads(l) for f in files for l in open(f)]).drop_duplicates(["key", "dataset", "units", "seed"])
else:          # records shipped with the repository
    rq = pd.read_csv(os.path.join(RQ, "results.csv"))
    rq["bits"] = rq.bits.map(lambda x: "inf" if x == float("inf") else str(int(x)))
res = pd.read_csv(MAIN)
res = res[res.kind.isin(["main", "ridge_bin"])]
print("RQ records:", len(rq), "| datasets:", rq.dataset.nunique(), "| seeds per key:", rq.groupby(["dataset", "units", "key"]).seed.nunique().min())
rq.to_csv(os.path.join(OUT, "results.csv"), index=False)

W = {"mm": "wisard_minmax", "qu": "wisard_quantile_uniform"}
rows = []
for (u, p), g in rq.groupby(["units", "pooling"]):
    e = res[(res.units == u) & (res.pooling == p)].groupby(["readout", "dataset"]).acc.mean()
    dss = sorted(g.dataset.unique())
    ridge = np.array([e[("ridge", ds)] for ds in dss])
    for q in ["mm", "qu"]:
        wis = np.array([e[(W[q], ds)] for ds in dss])
        for b in ["3", "4", "5", "6", "inf", "bsel"]:
            key = f"{p}|{q}|b={b}"
            a = g[g.key == key].groupby("dataset").acc.mean().reindex(dss).values
            pw = wilcoxon(a, wis).pvalue if np.any(a != wis) else 1.0
            rows.append(dict(units=u, pooling=p, quantizer=q, bits=b, ridge_q=a.mean(), ridge=ridge.mean(),
                             wisard=wis.mean(), gap_vs_wisard=(a - wis).mean(), wins_wisard=int((wis > a + 1e-9).sum()),
                             p_wilcoxon=pw, sel_bits=(g[g.key == key].bits.value_counts().to_dict() if b == "bsel" else "")))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(S.round(2).to_string(index=False))
