"""Diagnostics reported in the paper (13 benchmarks, mean-state pooling, N_h in {128, 1024}, reservoir seed 0):
  (i)  ridge-selected reservoir: number of principal components explaining 95% of the variance of the pooled states,
       and test accuracy of the ridge readout restricted to the top-k components (alpha by 5-fold CV on the training set);
  (ii) W-QU-selected reservoir and readout: training and test accuracy of W-QU and of the ridge readout on the same
       reservoir (alpha by 5-fold CV), i.e., how much each readout fits the training set.
Reads the selected configurations in results/main (best_*.json) and writes results/diagnostics/diagnostics.jsonl.
Usage: python diagnostics.py <datasets|all> <n_workers>"""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from sklearn.metrics import accuracy_score
from esnwisard import DATASETS, load_dataset, make_scaler
from esnw_fast import pooled, ridge_multi_alpha, FastWiSARD, wisard_predict
from model_selection import folds_of, ALPHAS, EXTRA, RES, NQ

OUT = os.environ.get("ESNW_DIAG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "diagnostics", "diagnostics.jsonl"))


def cv_alpha(P, y):
    val = {a: [] for a in ALPHAS}
    for tr, va in folds_of(y):
        for a, yp in ridge_multi_alpha(P[tr], y[tr], P[va], ALPHAS).items():
            val[a].append(accuracy_score(y[va], yp))
    return max(ALPHAS, key=lambda a: (np.mean(val[a]), -list(ALPHAS).index(a)))


def task(args):
    ds, units = args
    Xtr, ytr, Xte, yte = load_dataset(ds)
    best = json.load(open(os.path.join(RES, f"best_{ds}_N{units}.json")))
    rec = dict(dataset=ds, units=units, pooling="mean", seed=0)
    # (i) ridge-selected reservoir
    c = best["main|mean|ridge"]["cfg"]
    a = dict(units=units, sr=c["sr"], lr=c["lr"], input_scaling=c["input_scaling"], bias=c["bias"], seed=0)
    Ptr, Pte = pooled(Xtr, **a)["mean"], pooled(Xte, **a)["mean"]
    mu = Ptr.mean(0)
    _, s, Vt = np.linalg.svd(Ptr - mu, full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    rec["pcs95"] = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)
    rec["pr"] = float(1.0 / np.sum(ev ** 2))                      # participation ratio
    Rtr, Rte = (Ptr - mu) @ Vt.T, (Pte - mu) @ Vt.T
    for k0 in [4, 16, 64]:
        k = min(k0, Rtr.shape[1])
        al = cv_alpha(Rtr[:, :k], ytr)
        rec[f"ridge_pc{k0}"] = 100 * accuracy_score(yte, ridge_multi_alpha(Rtr[:, :k], ytr, Rte[:, :k], [al])[al])
    al = float(best["main|mean|ridge"]["params"][0].split("=")[1])
    rec["ridge_test"] = 100 * accuracy_score(yte, ridge_multi_alpha(Ptr, ytr, Pte, [al])[al])
    # (ii) W-QU-selected reservoir and readout
    b = best["main|mean|wisard_quantile_uniform"]; c = b["cfg"]
    hp = dict(x.split("=") for x in b["params"]); Q, n, bl = int(hp["Q"]), int(hp["n"]), hp["bl"] == "1"
    a = dict(units=units, sr=c["sr"], lr=c["lr"], input_scaling=c["input_scaling"], bias=c["bias"], seed=0)
    Ptr, Pte = pooled(Xtr, **a)["mean"], pooled(Xte, **a)["mean"]
    sc = make_scaler("quantile_uniform", n_quantiles=min(NQ, len(Ptr)))
    Z1, Z2 = sc.fit_transform(Ptr), sc.transform(Pte)
    w = FastWiSARD(n_bits=n, n_tics=Q, seed=0).fit(Z1, ytr)
    rec["wqu_train"] = 100 * accuracy_score(ytr, wisard_predict(w, w.responses(Z1), bl))
    rec["wqu_test"] = 100 * accuracy_score(yte, wisard_predict(w, w.responses(Z2), bl))
    al = cv_alpha(Ptr, ytr)
    yp = ridge_multi_alpha(Ptr, ytr, np.vstack([Ptr, Pte]), [al])[al]
    rec["ridge_same_train"] = 100 * accuracy_score(ytr, yp[:len(ytr)])
    rec["ridge_same_test"] = 100 * accuracy_score(yte, yp[len(ytr):])
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    from multiprocessing import Pool
    dss = DATASETS + EXTRA if sys.argv[1] == "all" else sys.argv[1].split(",")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tasks = [(ds, u) for u in [1024, 128] for ds in dss]
    with Pool(int(sys.argv[2])) as pool:
        for r in pool.imap_unordered(task, tasks):
            print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
