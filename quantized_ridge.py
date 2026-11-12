"""
Ridge readout on quantized reservoir states (the baseline for a readout that only sees digitized states, e.g. after an
ADC), with the same protocol as the main experiment (model_selection.py): N_h in {128, 1024}, last or mean state, 60 reservoir
points, stratified 5-fold CV x 3 seeds (100-102) on the training set, test on 5 seeds (0-4).

Every pooled feature is quantized with L = 2^b levels, b in {3, 4, 5, 6} (8..64 levels, the same range as Q for the
WiSARD readouts), by a quantizer fitted on the training data only:
    mm  uniform quantizer between training min and max (like an ADC whose range is set on the training data);
        same thresholds as W-MM
    qu  equal-frequency quantizer (QuantileTransformer with min(100, M) quantiles, as W-QU); same thresholds as W-QU
The ridge readout uses the reconstruction values (level + 0.5) / L in [0, 1]. References without quantization
(b = inf): min-max normalised states (mm) and quantile-transformed states (qu).
For every (pooling, quantizer, b) the reservoir and alpha in {1e-14, ..., 1e5} are selected separately; "bsel" also
selects b in {3, 4, 5, 6} by the same criterion.

Usage:  python quantized_ridge.py select|test|all [datasets|all] [n_workers]     (resumable)
        python quantized_ridge.py smoke
Environment: ESNW_DATA = folder with the UCR .ts files; ESNW_RQ_RES = output folder (default results/quantized_ridge).
"""
import os, sys, json, time, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from esnwisard import DATASETS, load_dataset, make_scaler
from esnw_fast import pooled, ridge_multi_alpha
from model_selection import configs, folds_of, UNITS, SEL_SEEDS, TEST_SEEDS, POOLINGS, COST_ORDER, EXTRA, NQ
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ESNW_RQ_RES", os.path.join(HERE, "results", "quantized_ridge"))
BITS = [3, 4, 5, 6]
QZ = ["mm", "qu"]
ALPHAS = np.logspace(-14, 5, 20)


class Quantizer:
    def __init__(self, kind, b):
        self.kind, self.b = kind, b

    def fit(self, X):
        if self.kind == "mm":
            self.mn = X.min(0); r = X.max(0) - self.mn; r[r == 0] = 1.0; self.r = r
        else:
            self.sc = make_scaler("quantile_uniform", n_quantiles=min(NQ, len(X))).fit(X)
        return self

    def transform(self, X):
        Z = (X - self.mn) / self.r if self.kind == "mm" else self.sc.transform(X)
        if self.b is None:
            return Z
        L = 2 ** self.b
        return (np.clip(np.floor(Z * L), 0, L - 1) + 0.5) / L


def variants():
    return [(q, b) for q in QZ for b in BITS + [None]]


def vname(q, b):
    return f"{q}|b={'inf' if b is None else b}"


def sel_path(ds, units, cid, seed):
    d = os.path.join(RES, "select", ds, f"N{units}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"cfg{cid:02d}_s{seed}.json")


def select_task(args):
    ds, units, cfg, seed = args
    path = sel_path(ds, units, cfg["id"], seed)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    Xtr, ytr, _, _ = load_dataset(ds)
    F = pooled(Xtr, units, cfg["sr"], cfg["lr"], cfg["input_scaling"], cfg["bias"], seed)
    scores = {}
    for pooling in POOLINGS:
        P = F[pooling]
        for tr, va in folds_of(ytr):
            for q, b in variants():
                qz = Quantizer(q, b).fit(P[tr])
                for a, yp in ridge_multi_alpha(qz.transform(P[tr]), ytr[tr], qz.transform(P[va]), ALPHAS).items():
                    scores.setdefault(f"{pooling}|{vname(q, b)}|alpha={a:.0e}", []).append(float(accuracy_score(ytr[va], yp)))
    rec = dict(dataset=ds, units=units, cfg=cfg, seed=seed, time=time.time() - t0, scores=scores)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)
    return path, rec["time"]


def aggregate(ds, units):
    """best[<pooling>|<q>|b=<b>] and best[<pooling>|<q>|b=bsel] -> dict(cfg, b, alpha, val_acc)."""
    table = {}
    for c in configs():
        for seed in SEL_SEEDS:
            p = sel_path(ds, units, c["id"], seed)
            if not os.path.exists(p):
                return None
            for k, v in json.load(open(p))["scores"].items():
                table.setdefault((c["id"], k), []).extend(v)
    cfgs = {c["id"]: c for c in configs()}
    best = {}

    def upd(key, cid, b, alpha, v):
        m = 100 * float(np.mean(v))
        if key not in best or m > best[key]["val_acc"] + 1e-9:
            best[key] = dict(cfg=cfgs[cid], b=b, alpha=alpha, val_acc=m, n_val=len(v))

    for (cid, k), v in table.items():
        pooling, q, bs, al = k.split("|")
        b, alpha = bs.split("=")[1], float(al.split("=")[1])
        upd(f"{pooling}|{q}|b={b}", cid, b, alpha, v)
        if b != "inf":
            upd(f"{pooling}|{q}|b=bsel", cid, b, alpha, v)
    assert all(x["n_val"] == 5 * len(SEL_SEEDS) for x in best.values()), "incomplete validation scores"
    with open(os.path.join(RES, f"best_{ds}_N{units}.json"), "w") as f:
        json.dump(best, f, indent=1)
    return best


def test_task(args):
    ds, units, seed = args
    out = os.path.join(RES, f"test_{ds}_N{units}_seed{seed}.jsonl")
    done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
    best = json.load(open(os.path.join(RES, f"best_{ds}_N{units}.json")))
    Xtr, ytr, Xte, yte = load_dataset(ds)
    t0 = time.time()
    keys = [k for k in best if k not in done]
    for cid in sorted({best[k]["cfg"]["id"] for k in keys}):
        cfg = next(best[k]["cfg"] for k in keys if best[k]["cfg"]["id"] == cid)
        a = dict(units=units, sr=cfg["sr"], lr=cfg["lr"], input_scaling=cfg["input_scaling"], bias=cfg["bias"], seed=seed)
        Ftr, Fte = pooled(Xtr, **a), pooled(Xte, **a)
        for key in [k for k in keys if best[k]["cfg"]["id"] == cid]:
            bb = best[key]
            pooling, q, _ = key.split("|")
            b = None if bb["b"] == "inf" else int(bb["b"])
            qz = Quantizer(q, b).fit(Ftr[pooling])
            yp = ridge_multi_alpha(qz.transform(Ftr[pooling]), ytr, qz.transform(Fte[pooling]), [bb["alpha"]])[bb["alpha"]]
            rec = dict(key=key, dataset=ds, units=units, seed=seed, pooling=pooling, quantizer=q, bits=bb["b"],
                       bsel=key.endswith("bsel"), alpha=bb["alpha"], cfg=cfg, val_acc=bb["val_acc"],
                       acc=100 * accuracy_score(yte, yp), f1=100 * f1_score(yte, yp, average="macro"))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return out, time.time() - t0


def run(stage, dss, nw):
    from multiprocessing import Pool
    dss = sorted(dss, key=lambda d: COST_ORDER.index(d) if d in COST_ORDER else len(COST_ORDER))
    if stage in ("select", "all"):
        tasks = [(ds, u, c, s) for u in sorted(UNITS, reverse=True) for ds in dss for c in configs() for s in SEL_SEEDS
                 if not os.path.exists(sel_path(ds, u, c["id"], s))]
        print(f"{len(tasks)} selection tasks on {nw} workers", flush=True)
        t0 = time.time()
        with Pool(nw, maxtasksperchild=20) as pool:
            for i, (p, dt) in enumerate(pool.imap_unordered(select_task, tasks, chunksize=1)):
                print(f"[{i+1}/{len(tasks)}] {os.path.relpath(p, RES)} {dt:.0f}s (elapsed {time.time()-t0:.0f}s)", flush=True)
    if stage in ("test", "all"):
        for ds in dss:
            for u in UNITS:
                assert aggregate(ds, u) is not None, f"selection incomplete for {ds} N={u}"
        tasks = [(ds, u, s) for u in sorted(UNITS, reverse=True) for ds in dss for s in TEST_SEEDS]
        print(f"{len(tasks)} test tasks on {nw} workers", flush=True)
        t0 = time.time()
        with Pool(nw, maxtasksperchild=4) as pool:
            for p, dt in pool.imap_unordered(test_task, tasks, chunksize=1):
                print(f"done {os.path.relpath(p, RES)} {dt:.0f}s (elapsed {time.time()-t0:.0f}s)", flush=True)


def smoke():
    import tempfile, model_selection as ms
    global RES
    RES = tempfile.mkdtemp(prefix="rqsmoke_")
    ms.N_RANDOM = 2
    t0 = time.time()
    for c in configs():
        for s in SEL_SEEDS:
            select_task(("ItalyPowerDemand", 128, c, s))
    best = aggregate("ItalyPowerDemand", 128)
    test_task(("ItalyPowerDemand", 128, 0))
    recs = [json.loads(l) for l in open(os.path.join(RES, "test_ItalyPowerDemand_N128_seed0.jsonl"))]
    print(f"smoke OK in {time.time() - t0:.0f}s: {len(best)} selections, {len(recs)} test records;",
          {r["key"]: round(r["acc"], 1) for r in recs})


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "smoke":
        smoke()
        sys.exit(0)
    dss = DATASETS + EXTRA if len(sys.argv) < 3 or sys.argv[2] == "all" else sys.argv[2].split(",")
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 2)
    os.makedirs(RES, exist_ok=True)
    run(stage, dss, nw)
