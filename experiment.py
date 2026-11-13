"""
ESN-WiSARD experiment of the paper: reservoir of N_h = 1024 units, mean-state pooling, 6 UCR benchmarks.

Benchmarks: the UCR datasets of our initial pool for which temporal alignment matters, i.e., for which 1-NN DTW with
the learned warping window improves over 1-NN Euclidean distance by at least 5% (UCR archive summary):
Earthquakes, FordA, MedicalImages, SyntheticControl, Trace, Yoga.

Reservoir (leaky ESN, 10% recurrent and input connectivity, state reset for every sequence): full grid of
    spectral radius {0.3, 0.6, 0.9, 1.0, 1.2} x input scaling {0.1, 0.5, 1, 2} x leak rate {0.01, 0.03, 0.1, 0.3, 0.5, 1}
    x bias range {0, 0.1, 1}  ->  360 configurations.
Readouts, each selected jointly with the reservoir (i.e., every readout has its own reservoir):
    ridge      ridge regression on the states, alpha in {1e-14, ..., 1e3}
    ridge_q    ridge regression on the states quantized with b in {3, 4, 5, 6} bits per feature, by a uniform quantizer
               (thresholds of W-MM, key 'mm') or an equal-frequency one (thresholds of W-QU, key 'qu'), b and
               alpha in {1e-14, ..., 1e5} selected
    W-MM, W-QU, W-QG   WiSARD on the thermometer code of the states scaled by min-max, uniform quantile or Gaussian
               quantile scaling; Q in {8, 16, 32, 64}, n in {8, 16, 32}, bleaching on/off
    W-QG-h     W-QG with hashed RAM nodes (n = 32, S = 256 16-bit counters per node), for the energy estimate
Control without reservoir ('raw'): the same ridge and WiSARD readouts on the raw series (every time step is a feature).
Selection: mean validation accuracy over stratified 5-fold CV of the training set (shuffle, random_state 42) x 3 seeds
(100, 101, 102) for the reservoir and the input-to-bits mapping; ties -> first in grid order.
Test: the selected configurations are retrained on the whole training set and tested with 10 fresh seeds (0..9).

Usage (resumable, tasks run in parallel):
    python experiment.py [select|test|all] [datasets|all] [n_workers]
    python experiment.py smoke
Environment: ESNW_DATA = folder with the UCR .ts files (default data/ts); ESNW_RES = output (default results).
"""
import os, sys, json, time, itertools, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from esnwisard import load_dataset, make_scaler
from esnw_fast import pooled, ridge_multi_alpha, FastWiSARD, wisard_predict
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ESNW_RES", os.path.join(HERE, "results"))
DATASETS = ["FordA", "Yoga", "Earthquakes", "MedicalImages", "Trace", "SyntheticControl"]   # most expensive first
UNITS = 1024
SPACE = dict(sr=[0.3, 0.6, 0.9, 1.0, 1.2], input_scaling=[0.1, 0.5, 1.0, 2.0],
             lr=[0.01, 0.03, 0.1, 0.3, 0.5, 1.0], bias=[0.0, 0.1, 1.0])
SEL_SEEDS = [100, 101, 102]
TEST_SEEDS = list(range(10))
ALPHAS = np.logspace(-14, 3, 18)
ALPHAS_Q = np.logspace(-14, 5, 20)
BITS = [3, 4, 5, 6]
QS = [8, 16, 32, 64]
NS = [8, 16, 32]
HASH_N, HASH_S = 32, 256
NQ = 100
VARIANTS = {"W-MM": "minmax", "W-QU": "quantile_uniform", "W-QG": "quantile_gaussian"}


def configs():
    keys = list(SPACE)
    return [dict(id=i + 1, **dict(zip(keys, v))) for i, v in enumerate(itertools.product(*[SPACE[k] for k in keys]))]


def folds_of(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


class Quantizer:
    """b-bit quantizer fitted on the training data: 'mm' uniform between training min and max (thresholds of W-MM),
    'qu' equal-frequency (thresholds of W-QU); returns the reconstruction values (level + 0.5) / 2^b in [0, 1]."""
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
        L = 2 ** self.b
        return (np.clip(np.floor(Z * L), 0, L - 1) + 0.5) / L


def features(X, cfg, seed):
    """Mean-state pooling of the reservoir states, or the raw series for the control (cfg None)."""
    if cfg is None:
        return np.ascontiguousarray(X[:, :, 0])
    return pooled(X, UNITS, cfg["sr"], cfg["lr"], cfg["input_scaling"], cfg["bias"], seed)["mean"]


def evaluate(Ptr, ytr, Pva, yva, seed, scores, quantized=True):
    """Adds the validation accuracy of every readout configuration to scores (key -> list)."""
    def add(key, yp):
        scores.setdefault(key, []).append(float(accuracy_score(yva, yp)))
    for a, yp in ridge_multi_alpha(Ptr, ytr, Pva, ALPHAS).items():
        add(f"ridge|alpha={a:.0e}", yp)
    if quantized:
        for q in ("mm", "qu"):
            for b in BITS:
                qz = Quantizer(q, b).fit(Ptr)
                for a, yp in ridge_multi_alpha(qz.transform(Ptr), ytr, qz.transform(Pva), ALPHAS_Q).items():
                    add(f"ridge_q|q={q}|b={b}|alpha={a:.0e}", yp)
    for name, variant in VARIANTS.items():
        sc = make_scaler(variant, n_quantiles=min(NQ, len(Ptr)))
        Ztr, Zva = sc.fit_transform(Ptr), sc.transform(Pva)
        for Q in QS:
            for n in NS:
                w = FastWiSARD(n_bits=n, n_tics=Q, seed=seed).fit(Ztr, ytr)
                R = w.responses(Zva)
                for bl in (1, 0):
                    add(f"{name}|Q={Q}|n={n}|bl={bl}", wisard_predict(w, R, bool(bl)))
                del w, R
            if name == "W-QG":
                w = FastWiSARD(n_bits=HASH_N, n_tics=Q, seed=seed, table_size=HASH_S).fit(Ztr, ytr)
                R = w.responses(Zva)
                for bl in (1, 0):
                    add(f"W-QG-h|Q={Q}|n={HASH_N}|bl={bl}", wisard_predict(w, R, bool(bl)))
                del w, R


def sel_path(ds, cid, seed):
    d = os.path.join(RES, "select", ds)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{'raw' if cid is None else f'cfg{cid:03d}'}_s{seed}.json")


def select_task(args):
    ds, cfg, seed = args
    cid = None if cfg is None else cfg["id"]
    path = sel_path(ds, cid, seed)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    Xtr, ytr, _, _ = load_dataset(ds)
    P = features(Xtr, cfg, seed)
    scores = {}
    for tr, va in folds_of(ytr):
        evaluate(P[tr], ytr[tr], P[va], ytr[va], seed, scores, quantized=cfg is not None)
    with open(path + ".tmp", "w") as f:
        json.dump(dict(dataset=ds, cfg=cfg, seed=seed, time=time.time() - t0, scores=scores), f)
    os.replace(path + ".tmp", path)
    return path, time.time() - t0


def aggregate(ds):
    """best[<setting>|<readout>] with setting 'esn' (reservoir + readout) or 'raw' (control without reservoir).
    For ridge_q the keys are esn|ridge_q|mm and esn|ridge_q|qu (b selected) and esn|ridge_q|<q>|b=<b> (fixed b)."""
    best = {}

    def upd(key, cfg, params, v):
        m = 100 * float(np.mean(v))
        if key not in best or m > best[key]["val_acc"] + 1e-9:
            best[key] = dict(cfg=cfg, params=params, val_acc=m, n_val=len(v))

    for cfg in configs() + [None]:
        table = {}
        for seed in SEL_SEEDS:
            p = sel_path(ds, None if cfg is None else cfg["id"], seed)
            if not os.path.exists(p):
                return None
            for k, v in json.load(open(p))["scores"].items():
                table.setdefault(k, []).extend(v)
        setting = "raw" if cfg is None else "esn"
        for k, v in table.items():
            readout, *params = k.split("|")
            if readout == "ridge_q":
                hp = dict(p.split("=") for p in params)
                upd(f"{setting}|ridge_q|{hp['q']}", cfg, params, v)
                upd(f"{setting}|ridge_q|{hp['q']}|b={hp['b']}", cfg, params, v)
            else:
                upd(f"{setting}|{readout}", cfg, params, v)
    assert all(b["n_val"] == 5 * len(SEL_SEEDS) for b in best.values()), "incomplete validation scores"
    with open(os.path.join(RES, f"best_{ds}.json"), "w") as f:
        json.dump(best, f, indent=1)
    return best


def test_task(args):
    ds, seed = args
    out = os.path.join(RES, "test", f"{ds}_seed{seed}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
    best = json.load(open(os.path.join(RES, f"best_{ds}.json")))
    Xtr, ytr, Xte, yte = load_dataset(ds)
    C = len(np.unique(ytr))
    t0 = time.time()
    keys = [k for k in best if k not in done]
    groups = {}
    for k in keys:
        c = best[k]["cfg"]
        groups.setdefault(None if c is None else c["id"], []).append(k)
    for cid, ks in groups.items():
        cfg = best[ks[0]]["cfg"]
        Ptr, Pte = features(Xtr, cfg, seed), features(Xte, cfg, seed)
        for key in ks:
            b = best[key]
            setting, readout = key.split("|")[:2]
            hp = dict(p.split("=") for p in b["params"])
            rec = dict(key=key, setting=setting, readout=readout, dataset=ds, seed=seed, cfg=cfg, params=hp,
                       val_acc=b["val_acc"], n_train=len(ytr), C=C, D=int(Ptr.shape[1]))
            if readout == "ridge":
                a = float(hp["alpha"])
                yp = ridge_multi_alpha(Ptr, ytr, Pte, [a])[a]
            elif readout == "ridge_q":
                a = float(hp["alpha"])
                qz = Quantizer(hp["q"], int(hp["b"])).fit(Ptr)
                yp = ridge_multi_alpha(qz.transform(Ptr), ytr, qz.transform(Pte), [a])[a]
            else:
                variant = VARIANTS[readout.replace("-h", "")]
                sc = make_scaler(variant, n_quantiles=min(NQ, len(Ptr)))
                Ztr, Zte = sc.fit_transform(Ptr), sc.transform(Pte)
                w = FastWiSARD(n_bits=int(hp["n"]), n_tics=int(hp["Q"]), seed=seed,
                               table_size=HASH_S if readout.endswith("-h") else None).fit(Ztr, ytr)
                yp = wisard_predict(w, w.responses(Zte), bool(int(hp["bl"])))
                rec.update(n_rams=w.A, n_bits_input=w.B, n_entries=int(w.n_entries_),
                           mem_kb_sparse=w.memory_sparse_bits() / 8 / 1024,
                           mem_kb_hashed16=(w.memory_hashed_bits(16) / 8 / 1024) if w.S else None)
            rec.update(acc=100 * accuracy_score(yte, yp), f1=100 * f1_score(yte, yp, average="macro"))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return out, time.time() - t0


def run(stage, dss, nw):
    from multiprocessing import Pool
    dss = sorted(dss, key=lambda d: DATASETS.index(d) if d in DATASETS else len(DATASETS))
    t0 = time.time()
    if stage in ("select", "all"):
        tasks = [(ds, c, s) for ds in dss for c in configs() + [None] for s in SEL_SEEDS
                 if not os.path.exists(sel_path(ds, None if c is None else c["id"], s))]
        print(f"{len(tasks)} selection tasks on {nw} workers", flush=True)
        with Pool(nw, maxtasksperchild=20) as pool:
            for i, (p, dt) in enumerate(pool.imap_unordered(select_task, tasks, chunksize=1)):
                print(f"[{i+1}/{len(tasks)}] {os.path.relpath(p, RES)} {dt:.0f}s (elapsed {time.time()-t0:.0f}s)", flush=True)
    if stage in ("test", "all"):
        for ds in dss:
            assert aggregate(ds) is not None, f"selection incomplete for {ds}"
        tasks = [(ds, s) for ds in dss for s in TEST_SEEDS]
        print(f"{len(tasks)} test tasks on {nw} workers", flush=True)
        with Pool(nw, maxtasksperchild=4) as pool:
            for p, dt in pool.imap_unordered(test_task, tasks, chunksize=1):
                print(f"done {os.path.relpath(p, RES)} {dt:.0f}s (elapsed {time.time()-t0:.0f}s)", flush=True)
        import pandas as pd
        recs = [json.loads(l) for ds in dss for s in TEST_SEEDS for l in open(os.path.join(RES, "test", f"{ds}_seed{s}.jsonl"))]
        df = pd.DataFrame(recs)
        df["cfg_id"] = df.cfg.apply(lambda c: None if c is None else c["id"])
        df.drop(columns=["cfg"]).to_csv(os.path.join(RES, "results.csv"), index=False)
        print(df.groupby(["dataset", "key"]).acc.mean().unstack().round(1).T.to_string(), flush=True)


def smoke():
    """Installation check (~1 min): 2 reservoir configurations on SyntheticControl, into a temporary folder."""
    import tempfile
    global RES, configs
    RES = tempfile.mkdtemp(prefix="esnw_smoke_")
    full = configs
    configs = lambda: full()[:2]
    run("all", ["SyntheticControl"], 4)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "smoke":
        smoke()
        sys.exit(0)
    dss = DATASETS if len(sys.argv) < 3 or sys.argv[2] == "all" else sys.argv[2].split(",")
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 2)
    os.makedirs(RES, exist_ok=True)
    run(stage, dss, nw)
