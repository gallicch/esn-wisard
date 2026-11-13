"""Download UCR datasets (aeon-toolkit .ts zips from timeseriesclassification.com) into <repo>/data and extract the
.ts files into $ESNW_DATA (default <repo>/data/ts). Usage: python fetch_ucr.py Name1,Name2,...  or  python fetch_ucr.py all"""
import os, sys, zipfile, urllib.request
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TS = os.environ.get("ESNW_DATA", os.path.join(DATA, "ts"))
os.makedirs(TS, exist_ok=True)
NAMES = ["Earthquakes", "FordA", "MedicalImages", "SyntheticControl", "Trace", "Yoga"]
for name in (NAMES if sys.argv[1] == "all" else sys.argv[1].split(",")):
    z = os.path.join(DATA, f"{name}.zip")
    if not os.path.exists(z):
        url = f"https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
        urllib.request.urlretrieve(url, z + ".part")
        os.replace(z + ".part", z)
    with zipfile.ZipFile(z) as f:
        names = [n for n in f.namelist() if n.endswith(".ts")]
        for n in names:
            with f.open(n) as src, open(os.path.join(TS, os.path.basename(n)), "wb") as dst:
                dst.write(src.read())
    print(name, "ok:", [os.path.basename(n) for n in names], flush=True)
