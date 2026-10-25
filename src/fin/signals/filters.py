"""Trend filters.

L1 trend filtering (Kim, Koh, Boyd & Gorinevsky, 2009):

    minimize  0.5 * ||y - x||_2^2 + lam * ||D x||_1

with D the second-difference operator. The L1 penalty makes D x sparse, so x is piecewise
linear: the kinks are the points where the trend changes. Above ``l1_trend_lam_max(y)`` the
solution is the least-squares straight line.

Applied to the last N points of a series, the filter is causal: x[-1] only uses data up to now.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solveh_banded


def second_difference(n: int) -> sp.csr_matrix:
    """(n-2) x n matrix with rows [.., 1, -2, 1, ..]."""
    return sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csr")


def _d(x):
    return x[:-2] - 2 * x[1:-1] + x[2:]


def _dt(z):
    out = np.zeros(len(z) + 2)
    out[:-2] += z
    out[1:-1] -= 2 * z
    out[2:] += z
    return out


def _ddt_bands(m: int) -> np.ndarray:
    """D D^T (pentadiagonal Toeplitz 1, -4, 6, -4, 1) in upper banded storage."""
    ab = np.zeros((3, m))
    ab[0, 2:] = 1.0
    ab[1, 1:] = -4.0
    ab[2, :] = 6.0
    return ab


def l1_trend_lam_max(y: np.ndarray) -> float:
    """Smallest lam for which the L1 trend of ``y`` is a single straight line."""
    y = np.asarray(y, dtype=float)
    return float(np.max(np.abs(solveh_banded(_ddt_bands(len(y) - 2), _d(y)))))


def l1_trend_filter(
    y: np.ndarray, lam: float, tol: float = 1e-8, max_iter: int = 100
) -> np.ndarray:
    """Piecewise-linear trend of ``y`` (see module docstring).

    Primal-dual interior-point method on the dual problem

        minimize 0.5 z' D D' z - y' D' z   subject to  -lam <= z <= lam,   x = y - D' z,

    following Kim et al.'s l1_tf. Each Newton step is one banded solve, so the cost is O(n) per
    iteration. ``tol`` is the duality gap relative to ``0.5 * ||y - mean(y)||^2``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError("need at least 3 points")
    m = n - 2
    alpha, beta, mu_factor = 0.01, 0.5, 2.0
    ddt = _ddt_bands(m)
    dy = _d(y)
    gap_tol = tol * max(0.5 * np.sum((y - y.mean()) ** 2), 1e-12)

    z = np.zeros(m)
    mu1, mu2 = np.ones(m), np.ones(m)
    f1, f2 = z - lam, -z - lam
    t, step = 1e-10, np.inf

    for _ in range(max_iter):
        dtz = _dt(z)
        ddtz = _d(dtz)
        w = dy - (mu1 - mu2)
        pobj1 = 0.5 * w @ solveh_banded(ddt, w) + lam * np.sum(mu1 + mu2)
        pobj2 = 0.5 * dtz @ dtz + lam * np.sum(np.abs(dy - ddtz))
        dobj = -0.5 * dtz @ dtz + dy @ z
        gap = min(pobj1, pobj2) - dobj
        if gap <= gap_tol:
            return y - dtz

        if step >= 0.2:
            t = max(2 * m * mu_factor / gap, 1.2 * t)

        # Newton step: (D D' - diag(mu1/f1 + mu2/f2)) dz = r
        s = ddt.copy()
        s[2] -= mu1 / f1 + mu2 / f2
        r = -ddtz + dy + (1 / t) / f1 - (1 / t) / f2
        dz = solveh_banded(s, r)
        dmu1 = -(mu1 + ((1 / t) + dz * mu1) / f1)
        dmu2 = -(mu2 + ((1 / t) - dz * mu2) / f2)

        residual = np.linalg.norm(np.concatenate([ddtz - w, -mu1 * f1 - 1 / t, -mu2 * f2 - 1 / t]))

        # Backtracking line search: stay strictly feasible and reduce the residual.
        step = 1.0
        for dmu, mu in ((dmu1, mu1), (dmu2, mu2)):
            neg = dmu < 0
            if neg.any():
                step = min(step, 0.99 * np.min(-mu[neg] / dmu[neg]))
        for _ in range(50):
            new_z = z + step * dz
            new_mu1, new_mu2 = mu1 + step * dmu1, mu2 + step * dmu2
            new_f1, new_f2 = new_z - lam, -new_z - lam
            new_residual = np.linalg.norm(
                np.concatenate(
                    [
                        _d(_dt(new_z)) - dy + new_mu1 - new_mu2,
                        -new_mu1 * new_f1 - 1 / t,
                        -new_mu2 * new_f2 - 1 / t,
                    ]
                )
            )
            if (
                max(new_f1.max(), new_f2.max()) < 0
                and new_residual <= (1 - alpha * step) * residual
            ):
                break
            step *= beta
        z, mu1, mu2, f1, f2 = new_z, new_mu1, new_mu2, new_f1, new_f2

    raise RuntimeError(f"L1 trend filter did not converge in {max_iter} iterations (gap {gap:.2e})")


def kinks(trend: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Indices where a piecewise-linear trend changes slope."""
    d2 = np.diff(np.asarray(trend), 2)
    scale = max(np.max(np.abs(np.diff(trend))), 1e-12)
    return np.flatnonzero(np.abs(d2) > tol * scale) + 1
