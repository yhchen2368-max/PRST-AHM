"""Limited-memory BFGS Hessian approximation.

1:1 Python translation of MRST LimitedMemoryHessian.m
"""

import numpy as np


class LimitedMemoryHessian:
    """Limited memory approximation of inverse Hessian (for L-BFGS).

    See: unitBoxBFGS (option limitedMemory)
    """

    def __init__(self, init_scale=1.0, init_strategy="static", m=5, sign=1):
        self.init_scale = init_scale
        self.init_strategy = init_strategy
        self.m = m
        self.sign = sign
        self.S = None  # stored control diffs (n x k)
        self.Y = None  # stored gradient diffs (n x k)
        self.it_count = 0
        self.nullspace = None

    def update(self, s, y):
        """Store new (s, y) vector pair."""
        s = np.asarray(s, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self.it_count += 1
        if self.it_count == 1:
            n = s.size
            if self.m > n:
                self.m = n
            self.S = np.zeros((n, self.m))
            self.Y = np.zeros((n, self.m))
            self.S[:, 0] = s
            self.Y[:, 0] = y
        else:
            if self.it_count <= self.m:
                self.S[:, self.it_count - 1] = s
                self.Y[:, self.it_count - 1] = y
            else:
                # Shift and replace oldest
                self.S[:, :-1] = self.S[:, 1:]
                self.Y[:, :-1] = self.Y[:, 1:]
                self.S[:, -1] = s
                self.Y[:, -1] = y
        return self

    def reset(self):
        self.S = None
        self.Y = None
        self.it_count = 0
        self.nullspace = None
        return self

    def apply_initial(self, v):
        v = np.asarray(v, dtype=float).ravel()
        if self.init_strategy == "static":
            return (self.sign * self.init_scale) * v
        elif self.init_strategy == "dynamic":
            s = self.S[:, self.it_count - 1] if self.it_count <= self.m else self.S[:, -1]
            y = self.Y[:, self.it_count - 1] if self.it_count <= self.m else self.Y[:, -1]
            return ((s.dot(y)) / (y.dot(y))) * v
        else:
            raise ValueError(f"Unknown strategy: {self.init_strategy}")

    def set_nullspace(self, Q=None):
        """Return a copy carrying nullspace Q.

        MATLAB's setNullspace has value semantics: `H = H.setNullspace(Q)`
        rebinds a copy and leaves the caller's Hessian untouched. That
        matters in projQ, which sets a nullspace for one projection only
        -- mutating in place would leak it into every later use of the
        same Hessian.
        """
        result = self.copy()
        if Q is None:
            result.nullspace = None
        else:
            Q = np.asarray(Q)
            if Q.dtype != bool:
                Q = Q.astype(float)
                if self.it_count > 0:
                    assert Q.shape[0] == self.S.shape[0], "Dimension mismatch"
                    assert Q.shape[0] >= Q.shape[1], (
                        "Number of columns in nullspace matrix exceeds "
                        "number of rows")
            result.nullspace = Q
        return result

    def active_pairs(self):
        """Return (S, Y) trimmed to the pairs actually stored.

        S and Y are preallocated to m columns here, so trailing columns
        are zero until m updates have happened -- unlike MATLAB, where
        they grow and `S(:,end)` is always the newest pair. Callers that
        want MATLAB's S and Y want these.
        """
        if self.S is None or self.Y is None or self.it_count == 0:
            return None, None
        k = min(self.it_count, self.m)
        return self.S[:, :k], self.Y[:, :k]

    def copy(self):
        result = LimitedMemoryHessian(
            init_scale=self.init_scale,
            init_strategy=self.init_strategy,
            m=self.m,
            sign=self.sign,
        )
        result.S = self.S.copy() if self.S is not None else None
        result.Y = self.Y.copy() if self.Y is not None else None
        result.it_count = self.it_count
        result.nullspace = self.nullspace
        return result

    def dot(self, v):
        """Multiply H * v (right-multiplication)."""
        v = np.asarray(v, dtype=float).ravel()
        if self.it_count == 0:
            r = (self.sign * self.init_scale) * v
            if self.nullspace is not None:
                if self.nullspace.dtype == bool:
                    r = r * (~self.nullspace)
                else:
                    r = r - self.nullspace @ (self.nullspace.T @ r)
            return r

        n_vec = min(self.it_count, self.m)
        rho = np.zeros(n_vec)
        alpha = np.zeros(n_vec)

        # First loop (reverse)
        for k in range(n_vec - 1, -1, -1):
            rho[k] = 1.0 / self.S[:, k].dot(self.Y[:, k])
            alpha[k] = rho[k] * self.S[:, k].dot(v)
            v = v - alpha[k] * self.Y[:, k]

        r = self.apply_initial(v)

        # Second loop (forward)
        for k in range(n_vec):
            beta = rho[k] * self.Y[:, k].dot(r)
            r = r + (alpha[k] - beta) * self.S[:, k]

        return r

    def __neg__(self):
        result = LimitedMemoryHessian(
            init_scale=self.init_scale,
            init_strategy=self.init_strategy,
            m=self.m,
            sign=-self.sign,
        )
        result.S = self.S.copy() if self.S is not None else None
        result.Y = self.Y.copy() if self.Y is not None else None
        result.it_count = self.it_count
        result.nullspace = self.nullspace
        return result

    def full(self):
        """Create full matrix (for debugging)."""
        if self.it_count == 0:
            n = 1
        else:
            n = self.S.shape[0]
        M = np.zeros((n, n))
        for k in range(n):
            ek = np.zeros(n)
            ek[k] = 1.0
            M[:, k] = self.dot(ek)
        return M
