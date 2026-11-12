"""
Main experiment: full model selection with last-state or mean-state pooling, for two separate reservoir sizes
(N_h = 128 and N_h = 1024), plus the ablations on the tuple size and on hashed RAM nodes, under the same protocol.

For every benchmark, N_h, pooling in {last, mean} and readout in {Ridge, W-MM, W-QU, W-QG}, the reservoir and the
readout hyper-parameters are selected jointly, separately for every readout, on the training set only.

Reservoir (reservoirpy defaults otherwise: 10% recurrent and input connectivity, Bernoulli +-input_scaling input
weights, Gaussian recurrent weights rescaled to sr, tanh, state reset for every sequence):
    sr in {0.3, 0.6, 0.9, 1.0, 1.2}, input_scaling in {0.1, 0.5, 1, 2}, leak rate in {0.01, 0.03, 0.1, 0.3, 0.5, 1},
    bias b in {0, 0.1, 1} (uniform in [-b, b])  -> 360 points; random search: 60 points sampled once (seed 2027),
    the same for all benchmarks, readouts and N_h.
Readouts (fixed grids):
    Ridge   alpha in {1e-14, 1e-13, ..., 1e3}
    WiSARD  Q in {8, 16, 32, 64}, n in {8, 16, 32}, bleaching on/off; n_q = 100 quantiles for QU/QG.
Criterion: mean validation accuracy over stratified 5-fold CV of the training set (shuffle, random_state 42) x 3
reservoir/mapping seeds (100, 101, 102); ties -> first in the order (configuration id, grid order).
Test: the selected configuration is retrained on the full training set, 5 fresh seeds (0..4).
Control Ridge_bin: ridge readout on the thermometer code of the selected W-QU (same reservoir, Q, scaling), alpha in
{1e-14, ..., 1e5} by the same 5-fold CV on the training set.
Ablations, same protocol and the same 60 reservoir points (record kinds in the results):
    e3  tuple size n in {4, 8, 16, 32} with exact counters (selection of reservoir, Q, bleaching for every fixed n),
    e4  hashed RAM nodes, n = 32, S in {16, 32, 64, 128, 256} counters per node, W-QU and W-QG.

Usage (resumable; tasks are independent and run in parallel with a process pool):
    python model_selection.py select [datasets|all] [n_workers]
    python model_selection.py test   [datasets|all] [n_workers]
    python model_selection.py all    [datasets|all] [n_workers]      # select + test
    python model_selection.py smoke                                   # 1-2 minute installation check
'all' = the 13 benchmarks of the paper. Environment: ESNW_DATA = folder with the UCR .ts files (default data/ts);
ESNW_RES = output folder (default results/main).
"""
import os, sys, json, time, itertools, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from esnwisard import DATASETS, load_dataset, make_scaler, thermometer_matrix
from esnw_fast import pooled, ridge_multi_alpha, linear_kernels, ridge_dual_multi_alpha, FastWiSARD, wisard_predict
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ESNW_RES", os.path.join(HERE, "results", "main"))

UNITS = [128, 1024]
SPACE = dict(sr=[0.3, 0.6, 0.9, 1.0, 1.2], input_scaling=[0.1, 0.5, 1.0, 2.0],
             lr=[0.01, 0.03, 0.1, 0.3, 0.5, 1.0], bias=[0.0, 0.1, 1.0])
N_RANDOM, SPACE_SEED = 60, 2027
SEL_SEEDS = [100, 101, 102]
TEST_SEEDS = [0, 1, 2, 3, 4]
POOLINGS = ["last", "mean"]
ALPHAS = np.logspace(-14, 3, 18)
ALPHAS_BIN = np.logspace(-14, 5, 20)
QS = [8, 16, 32, 64]
NS_MAIN = [8, 16, 32]
NS_ABL = [4, 8, 16, 32]
HASH_N, HASH_S = 32, [16, 32, 64, 128, 256]
NQ = 100
VARIANTS = {"wisard_minmax": "minmax", "wisard_quantile_uniform": "quantile_uniform",
            "wisard_quantile_gaussian": "quantile_gaussian"}
HASH_VARIANTS = ["wisard_quantile_uniform", "wisard_quantile_gaussian"]
# order used to schedule the tasks (most expensive first)
COST_ORDER = ["FordA", "Wafer", "ECG5000", "Yoga", "Earthquakes", "Strawberry", "ChlorineConcentration", "MedicalImages",
              "Trace", "SyntheticControl", "GunPoint", "ECG200", "ItalyPowerDemand"]
EXTRA = ["Earthquakes", "Strawberry", "ChlorineConcentration", "MedicalImages", "ECG200"]


def configs():
    """60 reservoir configurations sampled without replacement (fixed seed) from the 360-point grid."""
    keys = list(SPACE)
    grid = [dict(zip(keys, v)) for v in itertools.product(*[SPACE[k] for k in keys])]
    pick = sorted(np.random.RandomState(SPACE_SEED).choice(len(grid), N_RANDOM, replace=False))
    return [dict(id=i + 1, **grid[j]) for i, j in enumerate(pick)]


def folds_of(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


def sel_path(ds, units, cid, seed):
    d = os.path.join(RES, "select", ds, f"N{units}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"cfg{cid:02d}_s{seed}.json")


# ----------------------------------------------------------------------------
# Selection: validation accuracies of every readout configuration for one (dataset, N_h, reservoir config, seed)
# ----------------------------------------------------------------------------
def select_task(args):
    ds, units, cfg, seed = args
    path = sel_path(ds, units, cfg["id"], seed)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    Xtr, ytr, _, _ = load_dataset(ds)
    F = pooled(Xtr, units, cfg["sr"], cfg["lr"], cfg["input_scaling"], cfg["bias"], seed)
    folds = folds_of(ytr)
    scores = {}

    def add(key, acc):
        scores.setdefault(key, []).append(float(acc))

    for pooling in POOLINGS:
        P = F[pooling]
        for tr, va in folds:
            for a, yp in ridge_multi_alpha(P[tr], ytr[tr], P[va], ALPHAS).items():
                add(f"{pooling}|ridge|alpha={a:.0e}", accuracy_score(ytr[va], yp))
            for rname, variant in VARIANTS.items():
                sc = make_scaler(variant, n_quantiles=min(NQ, len(tr)))
                Ztr, Zva = sc.fit_transform(P[tr]), sc.transform(P[va])
                for Q in QS:
                    for n in NS_ABL:
                        w = FastWiSARD(n_bits=n, n_tics=Q, seed=seed).fit(Ztr, ytr[tr])
                        R = w.responses(Zva)
                        for bl in (1, 0):
                            add(f"{pooling}|{rname}|Q={Q}|n={n}|bl={bl}", accuracy_score(ytr[va], wisard_predict(w, R, bool(bl))))
                        del R
                        if n == HASH_N and rname in HASH_VARIANTS:
                            atr, ava = w.addresses(Ztr), w.addresses(Zva)
                            for S in HASH_S:
                                wh = FastWiSARD(n_bits=n, n_tics=Q, seed=seed, table_size=S).fit(Ztr, ytr[tr], addr=atr)
                                Rh = wh.responses(Zva, addr=ava)
                                for bl in (1, 0):
                                    add(f"{pooling}|{rname}|Q={Q}|n={n}|bl={bl}|S={S}",
                                        accuracy_score(ytr[va], wisard_predict(wh, Rh, bool(bl))))
                            del atr, ava
    rec = dict(dataset=ds, units=units, cfg=cfg, seed=seed, time=time.time() - t0, scores=scores)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)
    return path, rec["time"]


# ----------------------------------------------------------------------------
# Aggregation: best configuration for every (pooling, readout) of the main tables and of the ablations
# ----------------------------------------------------------------------------
def aggregate(ds, units):
    """Returns {key: dict(cfg, params, val_acc)} with keys
         main|<pooling>|<readout>                 readout hyper-parameters with n in NS_MAIN (exact counters)
         e3|<pooling>|<variant>|n=<n>             exact counters with fixed n in NS_ABL
         e4|<pooling>|<variant>|S=<S>             hashed RAM nodes (n = HASH_N) with fixed S
    Ties are resolved in favour of the first entry in the order (configuration id, grid order)."""
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

    def upd(key, cid, params, v):
        m = 100 * float(np.mean(v))
        if key not in best or m > best[key]["val_acc"] + 1e-9:
            best[key] = dict(cfg=cfgs[cid], params=params, val_acc=m, val_std=100 * float(np.std(v)), n_val=len(v))

    for (cid, k), v in table.items():
        pooling, readout, *params = k.split("|")
        if readout == "ridge":
            upd(f"main|{pooling}|ridge", cid, params, v)
            continue
        hp = dict(p.split("=") for p in params)
        if "S" in hp:
            upd(f"e4|{pooling}|{readout}|S={hp['S']}", cid, params, v)
            continue
        upd(f"e3|{pooling}|{readout}|n={hp['n']}", cid, params, v)
        if int(hp["n"]) in NS_MAIN:
            upd(f"main|{pooling}|{readout}", cid, params, v)
    assert all(b["n_val"] == 5 * len(SEL_SEEDS) for b in best.values()), "incomplete validation scores"
    with open(os.path.join(RES, f"best_{ds}_N{units}.json"), "w") as f:
        json.dump(best, f, indent=1)
    return best


# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------
def test_task(args):
    ds, units, seed = args
    out = os.path.join(RES, f"test_{ds}_N{units}_seed{seed}.jsonl")
    done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
    best = json.load(open(os.path.join(RES, f"best_{ds}_N{units}.json")))
    Xtr, ytr, Xte, yte = load_dataset(ds)
    C = len(np.unique(ytr))
    t0 = time.time()

    def log(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    keys = [k for k in best if k not in done]
    bin_keys = [f"ridge_bin|{p}" for p in POOLINGS if f"ridge_bin|{p}" not in done]
    states = {}
    for cid in sorted({best[k]["cfg"]["id"] for k in keys} | {best[f"main|{k.split('|')[1]}|wisard_quantile_uniform"]["cfg"]["id"] for k in bin_keys}):
        cfg = next(best[k]["cfg"] for k in best if best[k]["cfg"]["id"] == cid)
        a = dict(units=units, sr=cfg["sr"], lr=cfg["lr"], input_scaling=cfg["input_scaling"], bias=cfg["bias"], seed=seed)
        Ftr, Fte = pooled(Xtr, **a), pooled(Xte, **a)
        for key in [k for k in keys if best[k]["cfg"]["id"] == cid]:
            b = best[key]
            kind, pooling, readout = key.split("|")[:3]
            hp = {k: (float(v) if k == "alpha" else int(v)) for k, v in (p.split("=") for p in b["params"])}
            base = dict(key=key, kind=kind, dataset=ds, units=units, seed=seed, pooling=pooling, readout=readout, cfg=cfg,
                        val_acc=b["val_acc"], n_train=len(ytr), C=C, **hp)
            Ptr, Pte = Ftr[pooling], Fte[pooling]
            if readout == "ridge":
                yp = ridge_multi_alpha(Ptr, ytr, Pte, [hp["alpha"]])[hp["alpha"]]
                log(dict(base, acc=100 * accuracy_score(yte, yp), f1=100 * f1_score(yte, yp, average="macro"),
                         mem_kb_fp32=C * Ptr.shape[1] * 4 / 1024))
                continue
            sc = make_scaler(VARIANTS[readout], n_quantiles=min(NQ, len(Ptr)))
            Ztr, Zte = sc.fit_transform(Ptr), sc.transform(Pte)
            w = FastWiSARD(n_bits=hp["n"], n_tics=hp["Q"], seed=seed, table_size=hp.get("S")).fit(Ztr, ytr)
            yp = wisard_predict(w, w.responses(Zte), bool(hp["bl"]))
            log(dict(base, acc=100 * accuracy_score(yte, yp), f1=100 * f1_score(yte, yp, average="macro"),
                     n_rams=w.A, n_bits_input=w.B, n_entries=int(w.n_entries_),
                     mem_kb_sparse=w.memory_sparse_bits() / 8 / 1024, mem_kb_dense=w.memory_dense_bits() / 8 / 1024,
                     mem_kb_hashed16=(w.memory_hashed_bits(16) / 8 / 1024) if w.S else None))
        for bk in [k for k in bin_keys if best[f"main|{k.split('|')[1]}|wisard_quantile_uniform"]["cfg"]["id"] == cid]:
            # control: ridge readout on the thermometer code of the selected W-QU
            pooling = bk.split("|")[1]
            b = best[f"main|{pooling}|wisard_quantile_uniform"]
            Q = int(dict(p.split("=") for p in b["params"])["Q"])
            Ptr, Pte = Ftr[pooling], Fte[pooling]
            val = {a: [] for a in ALPHAS_BIN}
            for tr, va in folds_of(ytr):
                sc = make_scaler("quantile_uniform", n_quantiles=min(NQ, len(tr)))
                Btr = thermometer_matrix(sc.fit_transform(Ptr[tr]), Q); Bva = thermometer_matrix(sc.transform(Ptr[va]), Q)
                K, (Kva,) = linear_kernels(Btr, [Bva])
                for a, yp in ridge_dual_multi_alpha(K, Kva, ytr[tr], ALPHAS_BIN).items():
                    val[a].append(accuracy_score(ytr[va], yp))
            vbest = max(np.mean(v) for v in val.values())
            alpha = [a for a in ALPHAS_BIN if np.mean(val[a]) == vbest][0]          # first maximum in grid order
            sc = make_scaler("quantile_uniform", n_quantiles=min(NQ, len(Ptr)))
            Btr = thermometer_matrix(sc.fit_transform(Ptr), Q); Bte = thermometer_matrix(sc.transform(Pte), Q)
            K, (Kte,) = linear_kernels(Btr, [Bte])
            yp = ridge_dual_multi_alpha(K, Kte, ytr, [alpha])[alpha]
            log(dict(key=bk, kind="ridge_bin", dataset=ds, units=units, seed=seed, pooling=pooling, readout="ridge_on_binary",
                     cfg=b["cfg"], Q=Q, alpha=float(alpha), val_acc=100 * float(np.mean(val[alpha])), n_train=len(ytr), C=C,
                     acc=100 * accuracy_score(yte, yp), f1=100 * f1_score(yte, yp, average="macro")))
        del Ftr, Fte
    return out, time.time() - t0


# ----------------------------------------------------------------------------
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
    """Quick installation check (~1-2 min): 2 reservoir points, ItalyPowerDemand, N_h = 128, into a temporary folder."""
    import tempfile
    global RES, N_RANDOM
    RES, N_RANDOM = tempfile.mkdtemp(prefix="esnw_smoke_"), 2
    t0 = time.time()
    for c in configs():
        for s in SEL_SEEDS:
            select_task(("ItalyPowerDemand", 128, c, s))
    best = aggregate("ItalyPowerDemand", 128)
    test_task(("ItalyPowerDemand", 128, 0))
    recs = [json.loads(l) for l in open(os.path.join(RES, "test_ItalyPowerDemand_N128_seed0.jsonl"))]
    acc = {r["key"]: round(r["acc"], 1) for r in recs if r["kind"] in ("main", "ridge_bin")}
    print(f"smoke test OK in {time.time() - t0:.0f}s: {len(best)} selections, {len(recs)} test records; {acc}")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "smoke":
        smoke()
        sys.exit(0)
    dss = DATASETS + EXTRA if len(sys.argv) < 3 or sys.argv[2] == "all" else sys.argv[2].split(",")
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 2)
    os.makedirs(RES, exist_ok=True)
    run(stage, dss, nw)
