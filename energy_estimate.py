"""Analytical energy and memory estimate of the readouts (ridge vs WiSARD) for the selected configurations of the current
experiment, in a scenario where the reservoir is not digital (e.g. a physical reservoir): only the digital readout counts.

Energy per operation, 45 nm (M. Horowitz, "Computing's energy problem (and what we can do about it)", ISSCC 2014):
  fp32 mult 3.7 pJ, fp32 add 0.9 pJ, int8 mult 0.2 pJ, int32 add 0.1 pJ,
  SRAM read of 64 bits: 10 pJ (8 KB), 20 pJ (32 KB), 100 pJ (1 MB)  ->  E_acc(size) = 10 pJ * sqrt(size / 8 KB)
  (the square-root law matches the three points within 15%; below 8 KB it is an extrapolation).
Conventions (all favourable to ridge):
  ridge inference   C*D MACs + one read of every weight (fp32 or int8 weights and states);
  ridge training    compute only (Gram accumulation M*D(D+1)/2 + X'Y M*D*C + Cholesky D^3/6 + solves C*D^2 MACs, fp32);
                    the memory traffic of the D x D Gram matrix is ignored -> lower bound
  ridge update      recursive least squares, compute only: 2*D^2 + 2*C*D MACs -> lower bound
  WiSARD inference  C*A counter reads + C*A int32 adds (+ C*A for the bleaching histogram if bleaching is on);
                    address formation and thermometer code are wiring/comparators (not counted)
  WiSARD training   M*A read-modify-write of a counter (2 accesses + 1 add); update = A of them
  WiSARD memory     hashed RAM nodes (n = 32, S counters of 16 bits per node and class, ablation E4) -> fixed size,
                    stored either in one SRAM ("single") or in C*A small local tables ("local").
Digitization (one conversion per pooled feature) is common to both readouts and not counted.
Usage: python energy_estimate.py results/main/results.csv results/energy
"""
import sys, os, json
import numpy as np
import pandas as pd

FM, FA, IM8, IA32 = 3.7, 0.9, 0.2, 0.1


def e_acc(nbytes):
    return 10.0 * np.sqrt(nbytes / 8192.0)


d = pd.read_csv(sys.argv[1])
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rows = []
for (ds, units, pooling), g in d.groupby(["dataset", "units", "pooling"]):
    D = int(units)
    rid = g[(g.kind == "main") & (g.readout == "ridge")]
    C, M = int(rid.C.iloc[0]), int(rid.n_train.iloc[0])
    r = dict(dataset=ds, units=D, pooling=pooling, C=C, M=M, acc_ridge=rid.acc.mean())
    r["mem_ridge_model_kb"] = 4 * C * D / 1024
    r["mem_ridge_train_kb"] = (4 * D * D + 4 * C * D) / 1024
    r["E_inf_ridge_fp32_nJ"] = (C * D * (FM + FA) + C * D / 2 * e_acc(4 * C * D)) / 1e3
    r["E_inf_ridge_int8_nJ"] = (C * D * (IM8 + IA32) + C * D / 8 * e_acc(C * D)) / 1e3
    macs_train = M * D * (D + 1) / 2 + M * D * C + D ** 3 / 6 + C * D ** 2
    r["E_train_ridge_uJ"] = macs_train * (FM + FA) / 1e6
    r["E_upd_ridge_nJ"] = (2 * D * D + 2 * C * D) * (FM + FA) / 1e3
    for S in (128, 256):
        h = g[(g.kind == "e4") & (g.readout == "wisard_quantile_gaussian") & (g.S == S)]
        A, bl = int(h.n_rams.iloc[0]), int(h.bl.iloc[0])
        node = S * 2                                   # bytes per node and class (16-bit counters)
        tot = C * A * node
        r[f"acc_wqg_S{S}"] = h.acc.mean()
        r[f"A_S{S}"], r[f"Q_S{S}"] = A, int(h.Q.iloc[0])
        r[f"mem_wqg_S{S}_kb"] = tot / 1024
        for name, sz in (("local", node), ("single", tot)):
            r[f"E_inf_wqg_S{S}_{name}_nJ"] = C * A * (e_acc(sz) + IA32 * (1 + bl)) / 1e3
            r[f"E_train_wqg_S{S}_{name}_uJ"] = M * A * (2 * e_acc(sz) + IA32) / 1e6
            r[f"E_upd_wqg_S{S}_{name}_nJ"] = A * (2 * e_acc(sz) + IA32) / 1e3
    rows.append(r)
E = pd.DataFrame(rows)
E.to_csv(os.path.join(OUT, "energy_estimates.csv"), index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
ex = E[(E.dataset == "ECG5000") & (E.units == 1024) & (E.pooling == "mean")].iloc[0]
print("=== ECG5000, N_h = 1024, mean state ===")
for k, v in ex.items():
    print(f"  {k:28s} {v:.4g}" if isinstance(v, (float, np.floating)) else f"  {k:28s} {v}")

def gm(x):
    return float(np.exp(np.mean(np.log(x))))

print("\n=== ratios ridge / WiSARD (W-QG hashed, S = 256): geometric mean [min, max] over the 13 benchmarks ===")
summ = {}
for (u, p), g in E.groupby(["units", "pooling"]):
    s = {}
    for lab, num, den in [("inf fp32 / local", "E_inf_ridge_fp32_nJ", "E_inf_wqg_S256_local_nJ"),
                          ("inf int8 / local", "E_inf_ridge_int8_nJ", "E_inf_wqg_S256_local_nJ"),
                          ("inf fp32 / single", "E_inf_ridge_fp32_nJ", "E_inf_wqg_S256_single_nJ"),
                          ("train / local", "E_train_ridge_uJ", "E_train_wqg_S256_local_uJ"),
                          ("train / single", "E_train_ridge_uJ", "E_train_wqg_S256_single_uJ"),
                          ("update / local", "E_upd_ridge_nJ", "E_upd_wqg_S256_local_nJ"),
                          ("update / single", "E_upd_ridge_nJ", "E_upd_wqg_S256_single_nJ"),
                          ("train mem ridge / wisard", "mem_ridge_train_kb", "mem_wqg_S256_kb"),
                          ("model mem ridge / wisard", "mem_ridge_model_kb", "mem_wqg_S256_kb")]:
        x = g[num] / g[den]
        s[lab] = (round(gm(x), 3), round(x.min(), 3), round(x.max(), 3))
    s["acc ridge / W-QG S128 / W-QG S256"] = (round(g.acc_ridge.mean(), 1), round(g.acc_wqg_S128.mean(), 1),
                                             round(g.acc_wqg_S256.mean(), 1))
    summ[f"N{u} {p}"] = s
    print(f"-- N_h = {u}, {p}")
    for k, v in s.items():
        print(f"   {k:34s} {v}")
json.dump(summ, open(os.path.join(OUT, "energy_summary.json"), "w"), indent=1)
