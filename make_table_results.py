"""Per-task table of the main experiment on the 13 benchmarks: mean-state pooling for N_h = 128 and 1024 (Ridge, W-MM,
W-QU, W-QG, Ridge_bin), plus mean accuracy and average rank for mean-state and last-state pooling. No bold.
Usage: python make_table_results.py results/main/results.csv figures/tab_results.tex"""
import sys
import numpy as np
import pandas as pd

LEN = {"ECG5000": 140, "Wafer": 152, "GunPoint": 150, "FordA": 500, "Yoga": 426, "SyntheticControl": 60,
       "ItalyPowerDemand": 24, "Trace": 275, "ECG200": 96, "Strawberry": 235, "ChlorineConcentration": 166,
       "MedicalImages": 99, "Earthquakes": 512}
SHORT = {"SyntheticControl": "SynthControl", "ItalyPowerDemand": "ItalyPower", "ChlorineConcentration": "Chlorine"}
RD = ["ridge", "wisard_minmax", "wisard_quantile_uniform", "wisard_quantile_gaussian"]

d = pd.read_csv(sys.argv[1])
d = d[d.kind.isin(["main", "ridge_bin"])]
A = d.groupby(["pooling", "units", "readout", "dataset"]).acc.mean()
info = d.groupby("dataset")[["n_train", "C"]].first()
ORDER = sorted(LEN, key=lambda ds: (-info.loc[ds, "n_train"], ds))


def ranks(pooling, u):
    M = np.array([[A[(pooling, u, r, ds)] for r in RD] for ds in ORDER])
    return pd.DataFrame(-np.round(M, 6)).rank(axis=1, method="average").values.mean(0)   # exact ties stay ties


L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{1.7pt}",
     r"\renewcommand{\arraystretch}{0.92}",
     r"\begin{tabular}{lr|rrrrr|rrrrr}",
     r" & & \multicolumn{5}{c|}{$N_h = 128$} & \multicolumn{5}{c}{$N_h = 1024$}\\",
     r"\textbf{Benchmark} & $M$/$T$/$C$ & Ridge & W-MM & W-QU & W-QG & R$_{\mathrm{bin}}$ & Ridge & W-MM & W-QU & W-QG & R$_{\mathrm{bin}}$\\",
     r"\hline"]
for ds in ORDER:
    cells = []
    for u in [128, 1024]:
        cells += ["%.1f" % A[("mean", u, r, ds)] for r in RD + ["ridge_on_binary"]]
    L.append(f"{SHORT.get(ds, ds)} & {int(info.loc[ds, 'n_train'])}/{LEN[ds]}/{int(info.loc[ds, 'C'])} & " + " & ".join(cells) + r"\\")
L.append(r"\hline")
for pooling in ["mean", "last"]:
    m, rk = [], []
    for u in [128, 1024]:
        m += ["%.1f" % np.mean([A[(pooling, u, r, ds)] for ds in ORDER]) for r in RD + ["ridge_on_binary"]]
        rk += ["%.2f" % x for x in ranks(pooling, u)] + ["--"]
    L.append(r"\multicolumn{2}{l|}{mean, %s state} & " % pooling + " & ".join(m) + r"\\")
L[-1] = L[-1]
L += [r"\end{tabular}", r"\vspace{-1mm}",
      r"\caption{\footnotesize Test accuracy (\%, 5 seeds) with mean-state pooling ($M$ training sequences of length "
      r"$T$, $C$ classes), and mean accuracy for both poolings. R$_{\mathrm{bin}}$: ridge readout on the code of W-QU.}",
      r"\label{tab.results}", r"\vspace{-4mm}", r"\end{table}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print("\n".join(L))
