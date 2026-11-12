"""Figure 2 of the paper (mean-state pooling, averages over the 13 benchmarks):
(a, b) ridge readout on quantized states vs number of bits, N_h = 128 and 1024, with the WiSARD readouts that use the
       same thresholds (W-MM for the uniform quantizer, W-QU for the equal-frequency one);
(c)    tuple size n with exact counters: accuracy gap vs ridge;
(d)    hashed RAM nodes (n = 32, S counters of 16 bits): gap vs memory, with the exact counters (stars).
Usage: python make_figure_analysis.py results/main/results.csv results/quantized_ridge/results.csv figures/fig_analysis.pdf"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

res = pd.read_csv(sys.argv[1]); rq = pd.read_csv(sys.argv[2]); OUT = sys.argv[3]
COL = {"wisard_minmax": "#0072B2", "wisard_quantile_uniform": "#D55E00", "wisard_quantile_gaussian": "#009E73"}
MK = {"wisard_minmax": "o", "wisard_quantile_uniform": "s", "wisard_quantile_gaussian": "^"}
NAME = {"wisard_minmax": "W-MM", "wisard_quantile_uniform": "W-QU", "wisard_quantile_gaussian": "W-QG"}
P = "mean"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(1, 4, figsize=(6.3, 1.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))

ridge = res[(res.kind == "main") & (res.readout == "ridge") & (res.pooling == P)].groupby(["units", "dataset"]).acc.mean()
main = res[(res.kind == "main") & (res.pooling == P)].groupby(["units", "readout"]).acc.mean()
BITS = ["3", "4", "5", "6", "inf"]
X = np.arange(len(BITS))
for a, u in zip(ax[:2], [128, 1024]):
    g = rq[(rq.units == u) & (rq.pooling == P) & (~rq.bsel)]
    for q, w, ls in [("mm", "wisard_minmax", "-"), ("qu", "wisard_quantile_uniform", "--")]:
        y = [g[(g.quantizer == q) & (g.bits.astype(float) == float(b))].acc.mean() for b in BITS]
        a.plot(X, y, color="0.15", ls=ls, marker=MK[w], ms=3.2, lw=1.0, mfc="white" if q == "qu" else "0.15")
        a.axhline(main[(u, w)], color=COL[w], lw=1.2, ls=":" if q == "qu" else "-")
    a.axhline(main[(u, "ridge")], color="0.6", lw=0.8, ls="-.")
    a.set_xticks(X); a.set_xticklabels(["3", "4", "5", "6", r"$\infty$"])
    a.set_xlabel("bits per feature"); a.set_title(rf"$N_h = {u}$", fontsize=7, pad=2)
    a.set_ylim(79, 92)
ax[0].set_ylabel("test accuracy (%)")
ax[1].set_yticklabels([])

# (c) tuple size
e3 = res[(res.kind == "e3") & (res.pooling == P)]
for u, ls, fill in [(128, "--", False), (1024, "-", True)]:
    for r in COL:
        g = e3[(e3.units == u) & (e3.readout == r)].groupby(["n", "dataset"]).acc.mean().reset_index()
        g["gap"] = g.acc - g.dataset.map(lambda ds: ridge[(u, ds)])
        m = g.groupby("n").gap.mean()
        ax[2].plot(np.log2(m.index), m.values, color=COL[r], ls=ls, marker=MK[r], ms=3, lw=1.0, mfc=COL[r] if fill else "white")
ax[2].set_xticks([2, 3, 4, 5]); ax[2].set_xticklabels(["4", "8", "16", "32"])
ax[2].set_xlabel("tuple size $n$"); ax[2].set_ylabel("gap vs ridge (pp)"); ax[2].set_title("exact counters", fontsize=7, pad=2)
ax[2].axhline(0, color="0.6", lw=0.8, ls="-.")

# (d) hashed RAM nodes
e4 = res[(res.kind == "e4") & (res.pooling == P)]
gm = lambda x: float(np.exp(np.mean(np.log(x))))
for u, ls, fill in [(128, "--", False), (1024, "-", True)]:
    for r in ["wisard_quantile_uniform", "wisard_quantile_gaussian"]:
        g = e4[(e4.units == u) & (e4.readout == r)].groupby(["S", "dataset"]).agg(acc=("acc", "mean"), mem=("mem_kb_hashed16", "mean")).reset_index()
        g["gap"] = g.acc - g.dataset.map(lambda ds: ridge[(u, ds)])
        s = g.groupby("S").agg(gap=("gap", "mean"), mem=("mem", gm))
        ax[3].plot(s.mem, s.gap, color=COL[r], ls=ls, marker=MK[r], ms=3, lw=1.0, mfc=COL[r] if fill else "white")
        ex = res[(res.kind == "e3") & (res.pooling == P) & (res.units == u) & (res.readout == r) & (res.n == 32)]
        ex = ex.groupby("dataset").agg(acc=("acc", "mean"), mem=("mem_kb_sparse", "mean")).reset_index()
        ax[3].plot(gm(ex.mem), (ex.acc - ex.dataset.map(lambda ds: ridge[(u, ds)])).mean(), marker="*", ms=6,
                   color=COL[r], mfc=COL[r] if fill else "white", ls="none")
ax[3].set_xscale("log"); ax[3].set_xlabel("readout memory (KB)"); ax[3].set_title(r"hashed nodes, $n = 32$", fontsize=7, pad=2)
ax[3].axhline(0, color="0.6", lw=0.8, ls="-.")
for a in ax[2:]:
    a.set_ylim(-17, 1)
for a in ax:
    a.grid(True, color="0.9", lw=0.5); a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=6.5, length=2)
for a, l in zip(ax, "abcd"):
    a.text(-0.02, 1.07, f"({l})", transform=a.transAxes, fontsize=7, fontweight="bold", ha="right")

h = [Line2D([], [], color="0.15", marker="o", ms=3.2, lw=1.0, label="Ridge, uniform quant."),
     Line2D([], [], color="0.15", marker="s", ms=3.2, mfc="white", ls="--", lw=1.0, label="Ridge, quantile quant."),
     Line2D([], [], color="0.6", ls="-.", lw=0.8, label="Ridge"),
     Line2D([], [], color=COL["wisard_minmax"], marker="o", ms=3, lw=1.0, label="W-MM"),
     Line2D([], [], color=COL["wisard_quantile_uniform"], marker="s", ms=3, lw=1.0, label="W-QU"),
     Line2D([], [], color=COL["wisard_quantile_gaussian"], marker="^", ms=3, lw=1.0, label="W-QG"),
     Line2D([], [], color="0.4", ls="--", marker="o", mfc="white", ms=3, lw=1.0, label=r"$N_h=128$"),
     Line2D([], [], color="0.4", ls="-", marker="o", ms=3, lw=1.0, label=r"$N_h=1024$"),
     Line2D([], [], color="0.4", marker="*", ms=6, ls="none", label="exact counters")]
fig.legend(handles=h, loc="upper center", ncol=9, frameon=False, fontsize=6.1, handlelength=1.8, handletextpad=0.35,
           columnspacing=0.8, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(pad=0.3, w_pad=0.9, rect=(0.01, 0, 1, 0.89))
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=250)
print("ok")
