"""Fast components used by all the experiments:
- reservoir()/pooled(): reservoirpy initialisation (W, W_in, bias) and a batched NumPy run of the leaky ESN, with the
  state reset to zero for every sequence (states identical to reservoirpy's Reservoir.run up to 1e-16);
- ridge_multi_alpha(): RidgeClassifier semantics (+-1 targets, intercept) for many alphas with one SVD;
- ridge_dual_multi_alpha(): the same predictions computed from linear kernels (for very wide binary codes);
- FastWiSARD / wisard_predict(): the WiSARD of esnwisard.py with faster addressing, dense tables for n <= 8,
  bincount-built hashed tables and histogram-based bleaching (predictions identical to esnwisard.WiSARD).
"""
import numpy as np
from esnwisard import WiSARD


def reservoir(units, sr, lr, input_scaling, bias, seed, rc=0.1, inc=0.1):
    from reservoirpy.nodes import Reservoir
    from reservoirpy.mat_gen import uniform
    b = uniform(low=-bias, high=bias) if bias > 0 else 0.0
    r = Reservoir(units=units, sr=sr, lr=lr, input_scaling=input_scaling, rc_connectivity=rc,
                  input_connectivity=inc, bias=b, seed=seed)
    r.initialize(np.zeros((1, 1)))
    W = r.W.toarray() if hasattr(r.W, "toarray") else np.asarray(r.W)
    Win = r.Win.toarray() if hasattr(r.Win, "toarray") else np.asarray(r.Win)
    bvec = np.broadcast_to(np.asarray(r.bias, dtype=np.float64).ravel(), (units,)).copy()
    return W, Win, bvec


def pooled(X, units, sr, lr, input_scaling, bias, seed, bs=2048, rc=0.1, inc=0.1):
    """X: (M, T, 1) -> {'last': (M, N), 'mean': (M, N)}"""
    W, Win, b = reservoir(units, sr, lr, input_scaling, bias, seed, rc, inc)
    WT, WinT = W.T.copy(), Win.T.copy()
    M, T, _ = X.shape
    last = np.empty((M, units)); mean = np.empty((M, units))
    for i in range(0, M, bs):
        Xi = X[i:i + bs]
        S = np.zeros((len(Xi), units)); acc = np.zeros_like(S)
        for t in range(T):
            S = (1 - lr) * S + lr * np.tanh(S @ WT + Xi[:, t, :] @ WinT + b)
            acc += S
        last[i:i + bs] = S; mean[i:i + bs] = acc / T
    return {"last": last, "mean": mean}


def ridge_multi_alpha(Xtr, ytr, Xva, alphas):
    """Ridge classifier (sklearn RidgeClassifier semantics: +-1 targets, intercept) for many alphas via one SVD.
    Returns {alpha: predicted labels}."""
    classes = np.unique(ytr)
    if len(classes) == 2:
        Y = np.where(ytr == classes[1], 1.0, -1.0)[:, None]
    else:
        Y = np.where(ytr[:, None] == classes[None, :], 1.0, -1.0)
    xm, ym = Xtr.mean(0), Y.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    UtY = U.T @ (Y - ym)
    Zva = (Xva - xm) @ Vt.T
    out = {}
    for a in alphas:
        sc = Zva @ ((s / (s ** 2 + a))[:, None] * UtY) + ym
        out[a] = classes[(sc[:, 0] > 0).astype(int)] if len(classes) == 2 else classes[np.argmax(sc, 1)]
    return out


_BT_CACHE = {}   # thermometer codes reused across tuple sizes n (same data, same Q)


class FastWiSARD(WiSARD):
    """Same model as esnwisard.WiSARD; RAM addresses are computed with np.packbits (identical values, MSB first)."""

    def _addresses(self, X):
        M = X.shape[0]
        key = (id(X), self.Q, self.off_.tobytes(), self.rng_.tobytes())
        hit = _BT_CACHE.get(key)
        BT = hit[1] if hit is not None and hit[0] is X else None   # the cache keeps X alive: id cannot be reused
        if BT is None:
            L = self._thermo_levels(X)                                          # (M, D)
            # thermometer code, bit-major layout: row x = feature x // Q, threshold x % Q
            BT = (L.T[:, None, :] > np.arange(self.Q, dtype=np.int32)[None, :, None]).reshape(self.B, M)
            if len(_BT_CACHE) >= 4:
                _BT_CACHE.pop(next(iter(_BT_CACHE)))
            _BT_CACHE[key] = (X, BT)
        out = np.empty((self.A, M), dtype=np.uint64)
        nb = (self.n + 7) // 8
        pad = np.uint64(8 * nb - self.n)                                        # packbits pads the last byte with 0s
        step = max(1, int(5e7 // (M * self.n)))
        for a0 in range(0, self.A, step):
            bits = BT[self.sel_[a0:a0 + step]]                                  # (a, n, M) bool
            packed = np.packbits(bits, axis=1).astype(np.uint64)               # (a, nb, M), MSB first
            acc = packed[:, 0, :]
            for k in range(1, nb):
                acc = (acc << np.uint64(8)) | packed[:, k, :]
            out[a0:a0 + step] = acc >> pad
        return np.ascontiguousarray(out.T)

    def _fit_hashed(self, addr, y):
        """Same tables as esnwisard.WiSARD._fit_hashed, built with bincount instead of np.add.at."""
        rs = np.random.RandomState(self.seed + 1000)
        self.salt_ = rs.randint(0, 2**63 - 1, size=(self.k, self.A), dtype=np.int64).astype(np.uint64)
        self.tables_ = np.zeros((self.C, self.k, self.A, self.S), dtype=np.int32)
        H = self._hash(addr)                                                    # (k, M, A)
        base = (np.arange(self.A, dtype=np.int64) * self.S)[None, :]
        for c in range(self.C):
            for j in range(self.k):
                idx = (H[j, y == c] + base).ravel()
                self.tables_[c, j] = np.bincount(idx, minlength=self.A * self.S).reshape(self.A, self.S)
        self.n_entries_ = int((self.tables_ > 0).sum())
        return self

    def fit(self, X, y, addr=None):
        """As esnwisard.WiSARD.fit; `addr` (from addresses()) can be passed to reuse the RAM addresses across table
        sizes; exact counters are stored as dense tables when n <= 8 (same responses, same number of written cells)."""
        X = np.asarray(X, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.C = len(self.classes_)
        self.D = X.shape[1]
        self.B = self.D * self.Q
        self.A = int(np.ceil(self.B / self.n))
        mn, mx = X.min(axis=0), X.max(axis=0)
        rng = mx - mn
        rng[rng == 0] = 1.0
        if np.all(mn == 0) and np.all(rng == 1):
            self.off_, self.rng_ = np.zeros(self.D), np.ones(self.D)
        else:
            self.off_, self.rng_ = mn, rng
        rs = np.random.RandomState(self.seed)
        perm = rs.permutation(self.B) if self.mapping == "random" else np.arange(self.B)
        idx = (np.arange(self.A)[:, None] * self.n + np.arange(self.n)[None, :]) % self.B
        self.sel_ = perm[idx]
        if addr is None:
            addr = self._addresses(X)
        if self.S is not None:
            return self._fit_hashed(addr, y)
        if self.n <= 8:
            L = 1 << self.n
            base = (np.arange(self.A, dtype=np.int64) * L)[None, :]
            self.dense_ = np.stack([np.bincount((addr[y == c].astype(np.int64) + base).ravel(),
                                                minlength=self.A * L).reshape(self.A, L) for c in range(self.C)])
            self.dense_ = self.dense_.astype(np.int32)
            self.n_entries_ = int((self.dense_ > 0).sum())
            return self
        ramid = (np.arange(self.A, dtype=np.uint64) << np.uint64(32))[None, :]
        keys = ramid + addr
        self.keys_, self.counts_ = [], []
        self.n_entries_ = 0
        for c in range(self.C):
            u, cnt = np.unique(keys[y == c].ravel(), return_counts=True)
            self.keys_.append(u); self.counts_.append(cnt.astype(np.int32)); self.n_entries_ += len(u)
        return self

    def addresses(self, X):
        """RAM addresses of X with the mapping and scaling of the fitted model (fit() must have been called)."""
        return self._addresses(np.asarray(X, dtype=np.float64))

    def responses(self, X, chunk=1024, addr=None):
        X = np.asarray(X, dtype=np.float64)
        M = X.shape[0]
        R = np.zeros((M, self.C, self.A), dtype=np.int32)
        ram = np.arange(self.A)[None, :]
        for m0 in range(0, M, chunk):
            a = addr[m0:m0 + chunk] if addr is not None else self._addresses(X[m0:m0 + chunk])
            if self.S is not None:
                H = self._hash(a)
                for c in range(self.C):
                    R[m0:m0 + chunk, c] = np.min([self.tables_[c, j][ram, H[j]] for j in range(self.k)], axis=0)
            elif self.n <= 8:
                ai = a.astype(np.int64)
                for c in range(self.C):
                    R[m0:m0 + chunk, c] = self.dense_[c][ram, ai]
            else:
                keys = (np.arange(self.A, dtype=np.uint64) << np.uint64(32))[None, :] + a
                for c in range(self.C):
                    u, cnt = self.keys_[c], self.counts_[c]
                    pos = np.minimum(np.searchsorted(u, keys), len(u) - 1)
                    found = (u[pos] == keys) if len(u) > 0 else np.zeros_like(keys, dtype=bool)
                    R[m0:m0 + chunk, c] = np.where(found, cnt[pos], 0)
        return R


def wisard_predict(w, R, bleaching=True, chunk=256):
    """Vectorised equivalent of WiSARD.predict (same bleaching rule and tie-breaking) from responses R (M, C, A):
    the scores S_c(beta) for all thresholds are obtained from per-class histograms of the counter values, and beta is
    the first threshold at which the winner is unique with confidence >= w.conf (or the scores vanish -> beta = 1)."""
    part1 = (R >= 1).sum(axis=2)
    if not bleaching:
        return w.classes_[np.argmax(part1, axis=1)]
    M, C, A = R.shape
    out = np.empty(M, dtype=np.int64)
    for m0 in range(0, M, chunk):
        Rc = R[m0:m0 + chunk]
        m = Rc.shape[0]
        V = int(Rc.max()) if Rc.size else 0
        W = V + 2
        rows = np.arange(m * C, dtype=np.int64)[:, None] * W
        h = np.bincount((Rc.reshape(m * C, A).astype(np.int64) + rows).ravel(), minlength=m * C * W).reshape(m, C, W)
        ge = np.cumsum(h[:, :, ::-1], axis=2)[:, :, ::-1]          # ge[..., v] = #counters >= v
        P = ge[:, :, 1:V + 2]                                     # scores for beta = 1 .. V+1  (m, C, V+1)
        s = P.sum(1)
        mx = P.max(1)
        tie = (P == mx[:, None, :]).sum(1) > 1
        second = np.maximum(np.where(P < mx[:, None, :], P, -1).max(1), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            conf = np.where(tie | (mx == 0), 0.0, 1.0 - second / np.maximum(mx, 1))
        stop = (s == 0) | (conf >= w.conf)                        # always true at beta = V+1 (all scores zero)
        first = np.argmax(stop, axis=1)
        idx = np.arange(m)
        sel = P[idx, :, first]
        sel = np.where((s[idx, first] == 0)[:, None], part1[m0:m0 + chunk], sel)
        out[m0:m0 + chunk] = np.argmax(sel, axis=1)
    return w.classes_[out]


def linear_kernels(Xtr, Xev_list, chunk=4096):
    """Centered linear kernels K = Xc_tr Xc_tr^T and K_e = Xc_e Xc_tr^T (centering with the mean of Xtr), accumulated
    over column chunks in float64, so that very wide (binary) inputs never need a dense float64 copy."""
    Xtr = np.asarray(Xtr)
    M, D = Xtr.shape
    K = np.zeros((M, M))
    Ke = [np.zeros((len(X), M)) for X in Xev_list]
    for j in range(0, D, chunk):
        A = Xtr[:, j:j + chunk].astype(np.float64)
        mu = A.mean(0)
        A -= mu
        K += A @ A.T
        for i, X in enumerate(Xev_list):
            Ke[i] += (np.asarray(X[:, j:j + chunk], dtype=np.float64) - mu) @ A.T
    return K, Ke


def ridge_dual_multi_alpha(K, Ke, ytr, alphas):
    """Same predictions as ridge_multi_alpha (primal ridge with intercept, +-1 targets) from the centered kernels
    K (Mtr x Mtr) and Ke (Me x Mtr): scores = Ke (K + a I)^-1 Yc + mean(Y). Returns {alpha: labels}."""
    classes = np.unique(ytr)
    if len(classes) == 2:
        Y = np.where(ytr == classes[1], 1.0, -1.0)[:, None]
    else:
        Y = np.where(ytr[:, None] == classes[None, :], 1.0, -1.0)
    ym = Y.mean(0)
    lam, U = np.linalg.eigh(K)
    lam = np.maximum(lam, 0.0)
    UtY = U.T @ (Y - ym)
    KeU = Ke @ U
    out = {}
    for a in alphas:
        sc = KeU @ (UtY / (lam + a)[:, None]) + ym
        out[a] = classes[(sc[:, 0] > 0).astype(int)] if len(classes) == 2 else classes[np.argmax(sc, 1)]
    return out
