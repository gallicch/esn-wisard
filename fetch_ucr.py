"""Download the UCR benchmarks of the paper (aeon-toolkit .ts zips from timeseriesclassification.com) into <repo>/data
and write the .ts files into $ESNW_DATA (default <repo>/data/ts), with every series z-normalized as in the paper.
SemgHandMovementCh2 is not z-normalized in the UCR distribution: as for the results of the paper, every series of
ACSF1, FordB, RefrigerationDevices and SemgHandMovementCh2 is z-normalized here, x -> (x - mean(x)) / std(x), while
the other six benchmarks, already z-normalized in the UCR distribution, are written unchanged.
Usage: python fetch_ucr.py Name1,Name2,...  or  python fetch_ucr.py all"""
import os, sys, zipfile, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TS = os.environ.get("ESNW_DATA", os.path.join(DATA, "ts"))
os.makedirs(TS, exist_ok=True)
NAMES = ["ACSF1", "FordA", "FordB", "FreezerRegularTrain", "LargeKitchenAppliances", "RefrigerationDevices",
         "SemgHandMovementCh2", "SyntheticControl", "Trace", "TwoPatterns"]
ZNORM = {"ACSF1", "FordB", "RefrigerationDevices", "SemgHandMovementCh2"}


def znorm_line(line):
    s = line.strip()
    if not s or s.startswith(("#", "@")):
        return line.rstrip("\n")
    vals, lab = s.rsplit(":", 1)
    x = np.array(vals.split(","), dtype=float)
    sd = x.std()
    return ",".join(repr(float(v)) for v in (x - x.mean()) / (sd if sd > 0 else 1.0)) + ":" + lab


for name in (NAMES if sys.argv[1] == "all" else sys.argv[1].split(",")):
    z = os.path.join(DATA, f"{name}.zip")
    if not os.path.exists(z):
        url = f"https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
        urllib.request.urlretrieve(url, z + ".part")
        os.replace(z + ".part", z)
    with zipfile.ZipFile(z) as f:
        names = [n for n in f.namelist() if n.endswith(".ts")]
        for n in names:
            text = f.open(n).read().decode()
            if name in ZNORM:
                text = "\n".join(znorm_line(l) for l in text.splitlines()) + "\n"
            open(os.path.join(TS, os.path.basename(n)), "w").write(text)
            # check: per-series mean and standard deviation of the written file
            X = [np.array(l.strip().rsplit(":", 1)[0].split(","), dtype=float) for l in text.splitlines()
                 if l.strip() and not l.startswith(("#", "@"))]
            mu, sd = np.array([x.mean() for x in X]), np.array([x.std() for x in X])
            print(f"{os.path.basename(n)}: {len(X)} series, max |mean| {np.abs(mu).max():.1e}, std in "
                  f"[{sd.min():.3f}, {sd.max():.3f}]{' (z-normalized here)' if name in ZNORM else ''}", flush=True)
