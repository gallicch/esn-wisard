"""
ESN-WiSARD: reference implementation (data loading, reservoir, encodings, WiSARD readout).

Pipeline (following L. Russo's BSc thesis and code):
  time series -> ESN reservoir (reservoirpy) -> temporal pooling (last | mean | multi)
              -> scaling (minmax | quantile-uniform | quantile-gaussian)
              -> thermometer encoding (Q tics per feature)
              -> WiSARD readout (n-bit tuples, random mapping, integer counters, bleaching)

The WiSARD readout is a self-contained NumPy re-implementation that follows the
semantics of the C++ library used in the thesis (giordamaug/WisardClassifier-C_vectors):
  * thermometer code of each feature with Q bits: first floor(p*Q) bits set to 1
  * a fixed random permutation of the B = D*Q bits; RAM k reads bits
    perm[(k*n + i) % B], i = 0..n-1 (MSB first)
  * training = increment integer counter at the addressed cell of the discriminator
    of the target class (single pass)
  * inference = score_c(b) = #RAMs whose counter >= bleaching threshold b; b is
    raised from 1 until a unique winner emerges (confidence > 0.01); if raising b
    zeroes all scores, fall back to b = 1.
  * for inputs that are not already in [0,1] (gaussian quantile variant), the
    library rescales each feature with min/max of the training set.
"""
import os
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
DATA_DIR = os.environ.get("ESNW_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ts"))  # <name>_{TRAIN,TEST}.ts


def load_dataset(name):
    from aeon.datasets import load_from_ts_file
    from sklearn.preprocessing import LabelEncoder
    Xtr, ytr = load_from_ts_file(f"{DATA_DIR}/{name}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{DATA_DIR}/{name}_TEST.ts")
    le = LabelEncoder().fit(ytr)
    # (M, 1, T) -> (M, T, 1) as expected by reservoirpy
    Xtr = np.transpose(np.asarray(Xtr, dtype=np.float64), (0, 2, 1))
    Xte = np.transpose(np.asarray(Xte, dtype=np.float64), (0, 2, 1))
    return Xtr, le.transform(ytr), Xte, le.transform(yte)


# ----------------------------------------------------------------------------
# Reservoir + pooling
# ----------------------------------------------------------------------------
def reservoir_states(X, units, sr, lr, seed, rc_connectivity=0.1):
    """Run a reservoirpy ESN over a batch (M, T, D_in) of sequences, state reset
    between sequences. Returns (M, T, N)."""
    from reservoirpy.nodes import Reservoir
    res = Reservoir(units=units, sr=sr, lr=lr, rc_connectivity=rc_connectivity, seed=seed)
    out = []
    bs = 256
    for i in range(0, len(X), bs):
        if i > 0:
            res.reset()
        out.append(res.run(X[i:i + bs]))   # 3D input: state is reset for every sequence
    return np.concatenate(out, axis=0)


def pooled_states(X, units, sr, lr, seed, poolings=("multi", "last"), rc_connectivity=0.1, bs=128):
    """Memory-frugal variant: run the reservoir batch-wise and keep only the pooled
    features. Returns {pooling: (M, D_pool)}."""
    from reservoirpy.nodes import Reservoir
    res = Reservoir(units=units, sr=sr, lr=lr, rc_connectivity=rc_connectivity, seed=seed)
    out = {p: [] for p in poolings}
    for i in range(0, len(X), bs):
        if i > 0:
            res.reset()
        S = res.run(X[i:i + bs])
        for p in poolings:
            out[p].append(pool(S, p))
    return {p: np.concatenate(v, axis=0) for p, v in out.items()}


def pool(S, how):
    """S: (M, T, N) -> (M, D_pool)."""
    if how == "last":
        return S[:, -1, :]
    if how == "mean":
        return S.mean(axis=1)
    if how == "multi":
        return np.hstack([S.mean(axis=1), S.max(axis=1), S.std(axis=1)])
    raise ValueError(how)


# ----------------------------------------------------------------------------
# Scaling to [0,1] (continuous -> discrete interface)
# ----------------------------------------------------------------------------
def make_scaler(kind, n_quantiles=100, seed=42):
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "quantile_uniform":
        return make_pipeline(StandardScaler(),
                             QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform",
                                                 random_state=seed))
    if kind == "quantile_gaussian":
        return make_pipeline(StandardScaler(),
                             QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal",
                                                 random_state=seed))
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# WiSARD readout (NumPy)
# ----------------------------------------------------------------------------
class WiSARD:
    def __init__(self, n_bits=32, n_tics=16, seed=0, mapping="random", bleaching=True, conf=0.01,
                 table_size=None, n_hash=1):
        """table_size: if set (power of 2), every RAM node is a count-min sketch with n_hash rows of
        table_size counters, addressed by hashing the n-bit tuple (bounded, dense memory);
        if None, exact counters are kept for the written addresses (sparse storage)."""
        assert 1 <= n_bits <= 32
        self.n = n_bits
        self.S = table_size
        self.k = n_hash
        if table_size is not None:
            assert table_size & (table_size - 1) == 0, "table_size must be a power of 2"
        self.Q = n_tics
        self.seed = seed
        self.mapping = mapping
        self.bleaching = bleaching
        self.conf = conf

    # ---- encoding -------------------------------------------------------
    def _thermo_levels(self, X):
        """Number of active bits per feature, in {0..Q}, as the C++ library: floor(p*Q)
        after the library's own min-max rescaling with training statistics."""
        P = (X - self.off_) / self.rng_
        return np.floor(P * self.Q).astype(np.int32)

    def _addresses(self, X):
        """Return (M, A) uint64 addresses of all RAMs for all samples."""
        L = self._thermo_levels(X)                      # (M, D)
        M = L.shape[0]
        # bit index x in [0, B): feature x // Q, threshold x % Q; bit=1 iff (x % Q) < level
        feat = self.sel_ // self.Q                       # (A, n)
        thr = self.sel_ % self.Q                         # (A, n)
        addr = np.zeros((M, self.A), dtype=np.uint64)
        pw = (np.uint64(1) << np.arange(self.n - 1, -1, -1, dtype=np.uint64))  # MSB first
        # chunk over RAMs to bound memory
        step = max(1, int(2e7 // (M * self.n)))
        for a0 in range(0, self.A, step):
            f = feat[a0:a0 + step]                       # (a, n)
            t = thr[a0:a0 + step]
            bits = (L[:, f] > t[None, :, :])             # (M, a, n) bool
            addr[:, a0:a0 + step] = (bits.astype(np.uint64) * pw[None, None, :]).sum(axis=2, dtype=np.uint64)
        return addr

    # ---- training -------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.C = len(self.classes_)
        self.D = X.shape[1]
        self.B = self.D * self.Q
        self.A = int(np.ceil(self.B / self.n))
        # library behaviour: rescale with train min/max unless data already in [0,1]
        mn, mx = X.min(axis=0), X.max(axis=0)
        rng = mx - mn
        rng[rng == 0] = 1.0
        if np.all(mn == 0) and np.all(rng == 1):
            self.off_, self.rng_ = np.zeros(self.D), np.ones(self.D)
        else:
            self.off_, self.rng_ = mn, rng
        # bit mapping
        rs = np.random.RandomState(self.seed)
        perm = rs.permutation(self.B) if self.mapping == "random" else np.arange(self.B)
        idx = (np.arange(self.A)[:, None] * self.n + np.arange(self.n)[None, :]) % self.B
        self.sel_ = perm[idx]                            # (A, n) bit positions read by each RAM
        addr = self._addresses(X)                        # (M, A)
        if self.S is not None:
            return self._fit_hashed(addr, y)
        # counters: per class, sorted unique keys (ram << 32 | addr) and their counts
        ramid = (np.arange(self.A, dtype=np.uint64) << np.uint64(32))[None, :]
        keys = ramid + addr                              # (M, A) uint64  (n <= 32)
        self.keys_, self.counts_ = [], []
        self.n_entries_ = 0
        for c in range(self.C):
            kc = keys[y == c].ravel()
            u, cnt = np.unique(kc, return_counts=True)
            self.keys_.append(u)
            self.counts_.append(cnt.astype(np.int32))
            self.n_entries_ += len(u)
        return self

    # ---- hashed (count-min) RAM nodes ------------------------------------
    def _hash(self, addr):
        """addr (M, A) uint64 -> (k, M, A) int64 indices in [0, S)."""
        shift = np.uint64(64 - int(np.log2(self.S)))
        out = np.empty((self.k,) + addr.shape, dtype=np.int64)
        for j in range(self.k):
            h = (addr ^ self.salt_[j][None, :]) * np.uint64(0x9E3779B97F4A7C15)
            out[j] = (h >> shift).astype(np.int64)
        return out

    def _fit_hashed(self, addr, y):
        rs = np.random.RandomState(self.seed + 1000)
        self.salt_ = rs.randint(0, 2**63 - 1, size=(self.k, self.A), dtype=np.int64).astype(np.uint64)
        self.tables_ = np.zeros((self.C, self.k, self.A, self.S), dtype=np.int32)
        H = self._hash(addr)                              # (k, M, A)
        ram = np.arange(self.A)[None, :]
        for c in range(self.C):
            Hc = H[:, y == c, :]                          # (k, Mc, A)
            for j in range(self.k):
                np.add.at(self.tables_[c, j], (np.broadcast_to(ram, Hc[j].shape).ravel(), Hc[j].ravel()), 1)
        self.n_entries_ = int((self.tables_ > 0).sum())
        return self

    def memory_hashed_bits(self, counter_bits=16):
        """Dense count-min storage: C x k x A x S counters."""
        return self.C * self.k * self.A * self.S * counter_bits

    # ---- inference ------------------------------------------------------
    def responses(self, X, chunk=512):
        """Counter values at the addressed cells: (M, C, A) int32."""
        X = np.asarray(X, dtype=np.float64)
        M = X.shape[0]
        R = np.zeros((M, self.C, self.A), dtype=np.int32)
        if self.S is not None:
            ram = np.arange(self.A)[None, :]
            for m0 in range(0, M, chunk):
                H = self._hash(self._addresses(X[m0:m0 + chunk]))     # (k, m, A)
                for c in range(self.C):
                    est = np.min([self.tables_[c, j][ram, H[j]] for j in range(self.k)], axis=0)
                    R[m0:m0 + chunk, c, :] = est
            return R
        ramid = (np.arange(self.A, dtype=np.uint64) << np.uint64(32))[None, :]
        for m0 in range(0, M, chunk):
            keys = ramid + self._addresses(X[m0:m0 + chunk])     # (m, A)
            for c in range(self.C):
                u, cnt = self.keys_[c], self.counts_[c]
                pos = np.searchsorted(u, keys)
                pos = np.minimum(pos, len(u) - 1)
                found = (u[pos] == keys) if len(u) > 0 else np.zeros_like(keys, dtype=bool)
                R[m0:m0 + chunk, c, :] = np.where(found, cnt[pos], 0)
        return R

    def decision_function(self, X):
        R = self.responses(X)
        M = R.shape[0]
        out = np.zeros((M, self.C))
        for i in range(M):
            res = R[i]                                   # (C, A)
            if not self.bleaching:
                out[i] = (res >= 1).sum(axis=1)
                continue
            b = 1
            while True:
                part = (res >= b).sum(axis=1)
                if part.sum() == 0:
                    part = (res >= 1).sum(axis=1)
                    break
                mx = part.max()
                tie = (part == mx).sum() > 1
                second = part[part < mx].max() if (part < mx).any() else 0
                conf = 0.0 if tie else 1.0 - second / mx
                if conf >= self.conf:
                    break
                b += 1
            s = part.sum()
            out[i] = res.sum(axis=1) / self.A if s == 0 else part / s
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    # ---- cost model -----------------------------------------------------
    def memory_sparse_bits(self, counter_bits=16):
        """Content-addressable (sparse) storage: one (address, counter) entry per written cell."""
        return self.n_entries_ * (self.n + counter_bits)

    def memory_dense_bits(self, counter_bits=16):
        """Dense LUT storage: C discriminators x A RAMs x 2^n cells x counter."""
        return self.C * self.A * (2 ** self.n) * counter_bits


# ----------------------------------------------------------------------------
# Thermometer code as a float matrix (for the linear-on-binary control readout)
# ----------------------------------------------------------------------------
def thermometer_matrix(X01, Q):
    """X01 in [0,1] (M, D) -> (M, D*Q) float32 binary thermometer code."""
    L = np.floor(np.clip(X01, 0, 1) * Q).astype(np.int32)       # (M, D)
    thr = np.arange(Q)[None, None, :]
    return (L[:, :, None] > thr).reshape(X01.shape[0], -1).astype(np.float32)
