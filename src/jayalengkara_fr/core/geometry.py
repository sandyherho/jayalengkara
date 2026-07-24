"""
jayalengkara: Fisher-Rao geometric primitives.

Mathematical Foundations
-------------------------
This module implements closed-form primitives for the Fisher-Rao (information)
geometry of two elementary statistical families, together with the exponential-
family duality shared by both.

Univariate Gaussian family
    Coordinates (mu, sigma), sigma > 0. The Fisher-Rao metric is

        ds^2 = (d mu^2 + 2 d sigma^2) / sigma^2 ,

    which is the upper half-plane metric of constant curvature K = -1/2. Under
    the linear rescaling x = mu / sqrt(2) it becomes 2 (dx^2 + d sigma^2)/sigma^2,
    i.e. twice the standard Poincare half-plane metric in coordinates (x, sigma).
    All Gaussian computations are performed in the standard half-plane chart
    z = x + i y with x = mu / sqrt(2), y = sigma, and results are rescaled.

Categorical family
    The probability simplex with the Fisher metric ds^2 = sum_i d p_i^2 / p_i.
    The square-root embedding p -> 2 sqrt(p) is an isometry onto the radius-2
    sphere; the Fisher-Rao distance is 2 arccos(sum_i sqrt(p_i q_i)) and the
    constant curvature is K = +1/4.

Exponential-family duality
    Both families are exponential families and therefore carry a dually flat
    structure: natural coordinates theta and expectation coordinates eta, a
    convex log-partition psi(theta) and its Legendre conjugate phi(eta), and the
    canonical (Bregman) divergence that reduces to the Kullback-Leibler
    divergence between distributions.

Parallelism
    Grid-valued and batch-valued primitives are parallelized over their leading
    index with numba prange. Thread count is governed by the caller through
    configure_threads.
"""

import os
import numpy as np

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

    prange = range

    def set_num_threads(n):
        pass

    def get_num_threads():
        return 1


# Rescaling constant linking the Gaussian metric to the standard half-plane.
SQRT2 = np.sqrt(2.0)

# Curvatures of the two elementary families (closed form).
K_GAUSSIAN = -0.5
K_CATEGORICAL = 0.25


def configure_threads(n_cores):
    """
    Configure the numba thread pool.

    Parameters
    ----------
    n_cores : int or None
        Number of threads. None or 0 selects every available core.

    Returns
    -------
    n : int
        Number of threads actually requested.
    """
    if n_cores is None or n_cores == 0:
        n_cores = os.cpu_count() or 1
    n_cores = max(1, int(n_cores))
    if NUMBA_AVAILABLE:
        try:
            import numba
            n_cores = min(n_cores, int(numba.config.NUMBA_NUM_THREADS))
            set_num_threads(n_cores)
        except Exception:
            pass
    return n_cores


# =============================================================================
# GAUSSIAN FAMILY: half-plane chart and hyperbolic geometry
# =============================================================================

@njit(cache=True, fastmath=True)
def gauss_to_halfplane(mu, sigma):
    """Map Gaussian coordinates (mu, sigma) to half-plane (x, y)."""
    return mu / SQRT2, sigma


@njit(cache=True, fastmath=True)
def halfplane_to_gauss(x, y):
    """Map half-plane coordinates (x, y) back to Gaussian (mu, sigma)."""
    return x * SQRT2, y


@njit(cache=True, fastmath=True)
def hyperbolic_distance(x1, y1, x2, y2):
    """
    Riemannian distance in the standard Poincare half-plane.

    d(z1, z2) = arccosh(1 + |z1 - z2|^2 / (2 y1 y2)).
    """
    dx = x1 - x2
    dy = y1 - y2
    arg = 1.0 + (dx * dx + dy * dy) / (2.0 * y1 * y2)
    if arg < 1.0:
        arg = 1.0
    return np.arccosh(arg)


@njit(cache=True, fastmath=True)
def fisher_rao_distance_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Fisher-Rao distance between two univariate Gaussians.

    The Gaussian metric is twice the standard half-plane metric, so the distance
    is sqrt(2) times the hyperbolic distance in the (mu/sqrt(2), sigma) chart.
    """
    x1, y1 = gauss_to_halfplane(mu1, sigma1)
    x2, y2 = gauss_to_halfplane(mu2, sigma2)
    return SQRT2 * hyperbolic_distance(x1, y1, x2, y2)


@njit(parallel=True, cache=True, fastmath=True)
def apply_mobius_batch(a, b, c, d, X, Y):
    """
    Apply a real Mobius transformation z -> (a z + b) / (c z + d) to a batch of
    half-plane points given as arrays (X, Y). Returns transformed (Xo, Yo).
    """
    n = X.shape[0]
    Xo = np.empty(n, dtype=np.float64)
    Yo = np.empty(n, dtype=np.float64)
    for i in prange(n):
        zx = X[i]
        zy = Y[i]
        # numerator (a z + b)
        nx = a * zx + b
        ny = a * zy
        # denominator (c z + d)
        dx = c * zx + d
        dy = c * zy
        den = dx * dx + dy * dy
        if den < 1e-300:
            Xo[i] = zx
            Yo[i] = zy
        else:
            Xo[i] = (nx * dx + ny * dy) / den
            Yo[i] = (ny * dx - nx * dy) / den
    return Xo, Yo


@njit(cache=True, fastmath=True)
def halfplane_geodesic(x1, y1, x2, y2, n_samples):
    """
    Sample the minimizing geodesic between two half-plane points.

    Vertical geodesic when the abscissae coincide, otherwise the semicircle
    centred on the real axis through both points, parameterized by arc length.
    Returns arrays (xs, ys) of length n_samples.
    """
    xs = np.empty(n_samples, dtype=np.float64)
    ys = np.empty(n_samples, dtype=np.float64)
    if abs(x1 - x2) < 1e-12:
        # Vertical line: interpolate geometrically in y (constant-speed geodesic).
        lg1 = np.log(y1)
        lg2 = np.log(y2)
        for k in range(n_samples):
            t = k / (n_samples - 1.0)
            xs[k] = x1
            ys[k] = np.exp((1.0 - t) * lg1 + t * lg2)
        return xs, ys
    # Semicircle centre on real axis: xc where |z1 - xc| = |z2 - xc|.
    xc = ((x2 * x2 + y2 * y2) - (x1 * x1 + y1 * y1)) / (2.0 * (x2 - x1))
    r = np.sqrt((x1 - xc) * (x1 - xc) + y1 * y1)
    th1 = np.arctan2(y1, x1 - xc)
    th2 = np.arctan2(y2, x2 - xc)
    for k in range(n_samples):
        t = k / (n_samples - 1.0)
        th = (1.0 - t) * th1 + t * th2
        xs[k] = xc + r * np.cos(th)
        ys[k] = r * np.sin(th)
    return xs, ys


@njit(cache=True, fastmath=True)
def halfplane_to_disk(x, y):
    """Cayley transform of the upper half-plane onto the Poincare disk."""
    # w = (z - i) / (z + i)
    nx = x
    ny = y - 1.0
    dx = x
    dy = y + 1.0
    den = dx * dx + dy * dy
    if den < 1e-300:
        return 0.0, 0.0
    wx = (nx * dx + ny * dy) / den
    wy = (ny * dx - nx * dy) / den
    return wx, wy


@njit(parallel=True, cache=True, fastmath=True)
def halfplane_to_disk_batch(X, Y):
    """Batch Cayley transform of half-plane points onto the disk."""
    n = X.shape[0]
    U = np.empty(n, dtype=np.float64)
    V = np.empty(n, dtype=np.float64)
    for i in prange(n):
        U[i], V[i] = halfplane_to_disk(X[i], Y[i])
    return U, V


@njit(parallel=True, cache=True, fastmath=True)
def orbit_distance_field(gx, gy, ox, oy):
    """
    Minimum hyperbolic distance from each grid point to a set of orbit anchors.

    Parameters
    ----------
    gx, gy : (H, W) grids of half-plane coordinates.
    ox, oy : (P,) orbit anchor coordinates.

    Returns
    -------
    field : (H, W) array of minimum hyperbolic distances.
    """
    H, W = gx.shape
    P = ox.shape[0]
    field = np.empty((H, W), dtype=np.float64)
    for i in prange(H):
        for j in range(W):
            x = gx[i, j]
            y = gy[i, j]
            best = 1e300
            for k in range(P):
                dx = x - ox[k]
                dy = y - oy[k]
                arg = 1.0 + (dx * dx + dy * dy) / (2.0 * y * oy[k])
                if arg < 1.0:
                    arg = 1.0
                dist = np.arccosh(arg)
                if dist < best:
                    best = dist
            field[i, j] = best
    return field


# =============================================================================
# CATEGORICAL FAMILY: square-root embedding and spherical geometry
# =============================================================================

@njit(cache=True, fastmath=True)
def simplex_to_sphere(p):
    """Square-root embedding p -> 2 sqrt(p) onto the radius-2 sphere."""
    n = p.shape[0]
    s = np.empty(n, dtype=np.float64)
    for i in range(n):
        s[i] = 2.0 * np.sqrt(p[i])
    return s


@njit(cache=True, fastmath=True)
def sphere_to_simplex(s):
    """Inverse embedding (s/2)^2 back to the simplex."""
    n = s.shape[0]
    p = np.empty(n, dtype=np.float64)
    for i in range(n):
        p[i] = (s[i] / 2.0) ** 2
    return p


@njit(cache=True, fastmath=True)
def fisher_rao_distance_categorical(p, q):
    """
    Fisher-Rao distance between categorical distributions.

    d(p, q) = 2 arccos(sum_i sqrt(p_i q_i)), the geodesic (great-circle) distance
    on the radius-2 sphere.
    """
    acc = 0.0
    for i in range(p.shape[0]):
        acc += np.sqrt(p[i] * q[i])
    if acc > 1.0:
        acc = 1.0
    if acc < -1.0:
        acc = -1.0
    return 2.0 * np.arccos(acc)


@njit(cache=True, fastmath=True)
def slerp(s1, s2, t):
    """
    Spherical linear interpolation between two points on a sphere of common
    radius, at parameter t in [0, 1]. Constant-speed great-circle geodesic.
    """
    n = s1.shape[0]
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in range(n):
        dot += s1[i] * s2[i]
        n1 += s1[i] * s1[i]
        n2 += s2[i] * s2[i]
    r = np.sqrt(n1)
    cos_omega = dot / (np.sqrt(n1) * np.sqrt(n2))
    if cos_omega > 1.0:
        cos_omega = 1.0
    if cos_omega < -1.0:
        cos_omega = -1.0
    omega = np.arccos(cos_omega)
    out = np.empty(n, dtype=np.float64)
    if omega < 1e-9:
        for i in range(n):
            out[i] = (1.0 - t) * s1[i] + t * s2[i]
        return out
    s_omega = np.sin(omega)
    c1 = np.sin((1.0 - t) * omega) / s_omega
    c2 = np.sin(t * omega) / s_omega
    for i in range(n):
        out[i] = c1 * s1[i] + c2 * s2[i]
    # Renormalize to the common radius against slerp rounding.
    nn = 0.0
    for i in range(n):
        nn += out[i] * out[i]
    scale = r / np.sqrt(nn)
    for i in range(n):
        out[i] = out[i] * scale
    return out


@njit(cache=True, fastmath=True)
def geodesic_arc_sphere(s1, s2, n_samples):
    """Sample the great-circle arc between two sphere points (n_samples, dim)."""
    dim = s1.shape[0]
    arc = np.empty((n_samples, dim), dtype=np.float64)
    for k in range(n_samples):
        t = k / (n_samples - 1.0)
        pk = slerp(s1, s2, t)
        for j in range(dim):
            arc[k, j] = pk[j]
    return arc


@njit(cache=True, fastmath=True)
def shannon_entropy(p):
    """Shannon entropy H(p) = -sum p_i log p_i (natural log)."""
    h = 0.0
    for i in range(p.shape[0]):
        if p[i] > 1e-300:
            h -= p[i] * np.log(p[i])
    return h


@njit(parallel=True, cache=True, fastmath=True)
def entropy_field_ternary(A, B):
    """
    Shannon entropy over a ternary-coordinate grid.

    Parameters
    ----------
    A, B : (H, W) grids of barycentric coordinates p0 = A, p1 = B (p2 = 1-A-B).
        Points outside the simplex (A + B > 1) receive NaN.

    Returns
    -------
    field : (H, W) entropy values.
    """
    H, W = A.shape
    field = np.empty((H, W), dtype=np.float64)
    for i in prange(H):
        for j in range(W):
            p0 = A[i, j]
            p1 = B[i, j]
            p2 = 1.0 - p0 - p1
            if p0 < 0.0 or p1 < 0.0 or p2 < 0.0:
                field[i, j] = np.nan
            else:
                h = 0.0
                if p0 > 1e-300:
                    h -= p0 * np.log(p0)
                if p1 > 1e-300:
                    h -= p1 * np.log(p1)
                if p2 > 1e-300:
                    h -= p2 * np.log(p2)
                field[i, j] = h
    return field


# =============================================================================
# EXPONENTIAL-FAMILY DUALITY (Gaussian)
# =============================================================================

@njit(cache=True, fastmath=True)
def gauss_theta(mu, sigma):
    """Natural parameters theta = (mu/sigma^2, -1/(2 sigma^2))."""
    s2 = sigma * sigma
    return mu / s2, -1.0 / (2.0 * s2)


@njit(cache=True, fastmath=True)
def theta_to_gauss(t1, t2):
    """Invert natural parameters back to (mu, sigma)."""
    sigma = np.sqrt(-1.0 / (2.0 * t2))
    mu = t1 * sigma * sigma
    return mu, sigma


@njit(cache=True, fastmath=True)
def gauss_eta(mu, sigma):
    """Expectation parameters eta = (mu, mu^2 + sigma^2)."""
    return mu, mu * mu + sigma * sigma


@njit(cache=True, fastmath=True)
def eta_to_gauss(e1, e2):
    """Invert expectation parameters back to (mu, sigma)."""
    mu = e1
    var = e2 - e1 * e1
    if var < 1e-300:
        var = 1e-300
    return mu, np.sqrt(var)


@njit(cache=True, fastmath=True)
def kl_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Kullback-Leibler divergence D(p || q) between univariate Gaussians.

    D = log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2)/(2 sigma2^2) - 1/2.

    This is the canonical (Bregman) divergence of the Gaussian exponential
    family expressed in distribution parameters.
    """
    dm = mu1 - mu2
    return (np.log(sigma2 / sigma1)
            + (sigma1 * sigma1 + dm * dm) / (2.0 * sigma2 * sigma2)
            - 0.5)


@njit(parallel=True, cache=True, fastmath=True)
def kl_asymmetry_field(MU, SIG, mu0, sigma0):
    """
    Divergence asymmetry D(p || p0) - D(p0 || p) over a grid of Gaussians p.

    Parameters
    ----------
    MU, SIG : (H, W) grids of (mu, sigma).
    mu0, sigma0 : reference Gaussian p0.

    Returns
    -------
    field : (H, W) signed asymmetry.
    """
    H, W = MU.shape
    field = np.empty((H, W), dtype=np.float64)
    for i in prange(H):
        for j in range(W):
            mu = MU[i, j]
            sig = SIG[i, j]
            d_pq = kl_gaussian(mu, sig, mu0, sigma0)
            d_qp = kl_gaussian(mu0, sigma0, mu, sig)
            field[i, j] = d_pq - d_qp
    return field


@njit(cache=True, fastmath=True)
def fisher_inner_gaussian(mu, sigma, u1, u2, v1, v2):
    """
    Fisher inner product of two tangent vectors u, v at the Gaussian (mu, sigma).

    The Fisher metric of the univariate Gaussian in (mu, sigma) coordinates is
    diag(1/sigma^2, 2/sigma^2).
    """
    s2 = sigma * sigma
    return (u1 * v1) / s2 + 2.0 * (u2 * v2) / s2
