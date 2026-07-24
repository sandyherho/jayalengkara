"""
jayalengkara: generators for the four Fisher-Rao cases.

Each generator produces the raw geometry consumed by the engine in models.py.
The four cases are deliberately elementary and closed form so that the geometry
is exact and the visual content is a faithful rendering of the manifold rather
than an approximation.

Case 1  Gaussian manifold as the hyperbolic plane, tiled by the modular group.
Case 2  Categorical manifold as the sphere, tiled by geodesic subdivision.
Case 3  The dually flat weave of the Gaussian exponential family.
Case 4  Brownian motion of Gaussians under the Fisher-Rao metric.
"""

import numpy as np

from . import geometry as geo

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

    prange = range


# =============================================================================
# CASE 1: modular tessellation of the hyperbolic plane
# =============================================================================

def _psl_key(M):
    """Canonical PSL(2, Z) key: identify M with -M by sign-fixing."""
    a, b, c, d = int(M[0, 0]), int(M[0, 1]), int(M[1, 0]), int(M[1, 1])
    for v in (a, c, b, d):
        if v != 0:
            if v < 0:
                a, b, c, d = -a, -b, -c, -d
            break
    return (a, b, c, d)


def modular_orbit(max_tiles=160):
    """
    Generate elements of the modular group PSL(2, Z) by breadth-first word
    enumeration over the generators S and T (and T inverse).

    Exact integer arithmetic is used throughout; floating point enters only when
    the transformations are later applied to sample points. This removes the
    dominant numerical hazard of hyperbolic tiling, which is precision loss in
    products of near-boundary transformations.

    Returns
    -------
    mats : (P, 2, 2) int64 array of distinct group elements (identity first).
    """
    S = np.array([[0, -1], [1, 0]], dtype=np.int64)
    T = np.array([[1, 1], [0, 1]], dtype=np.int64)
    Ti = np.array([[1, -1], [0, 1]], dtype=np.int64)
    gens = [S, T, Ti]

    I = np.eye(2, dtype=np.int64)
    seen = {_psl_key(I)}
    order = [I]
    frontier = [I]

    while frontier and len(order) < max_tiles:
        nxt = []
        for M in frontier:
            for g in gens:
                Mg = g @ M
                key = _psl_key(Mg)
                if key not in seen:
                    seen.add(key)
                    order.append(Mg)
                    nxt.append(Mg)
                    if len(order) >= max_tiles:
                        break
            if len(order) >= max_tiles:
                break
        frontier = nxt

    return np.array(order, dtype=np.int64)


def fundamental_domain_sides(n_samples=48, y_top=3.0):
    """
    Sample the three sides of the standard modular fundamental domain in the
    upper half-plane: the vertical segments x = -1/2 and x = 1/2 and the unit
    circular arc between angles 60 and 120 degrees.

    Returns
    -------
    sides : list of (n_samples, 2) polylines in half-plane (x, y) coordinates.
    """
    y_bot = np.sqrt(1.0 - 0.25)  # where |z| = 1 meets x = 1/2
    ys = np.linspace(y_bot, y_top, n_samples)
    left = np.column_stack([np.full(n_samples, -0.5), ys])
    right = np.column_stack([np.full(n_samples, 0.5), ys])
    th = np.linspace(np.pi / 3.0, 2.0 * np.pi / 3.0, n_samples)
    arc = np.column_stack([np.cos(th), np.sin(th)])
    return [left, right, arc]


def tessellation_geometry(max_tiles=160, n_samples=48, anchor=(0.0, 1.4)):
    """
    Build the disk-coordinate geometry of the modular tessellation.

    Returns
    -------
    edges_disk : (E, n_samples, 2) tile-side polylines in Poincare disk coords.
    anchors_disk : (P, 2) tile anchor points in disk coordinates.
    anchors_gauss : (P, 2) the same anchors in Gaussian (mu, sigma) coordinates.
    anchor_scale : (P,) local hyperbolic scale (anchor y in half-plane).
    mats : (P, 2, 2) integer group elements.
    """
    mats = modular_orbit(max_tiles)
    sides = fundamental_domain_sides(n_samples)

    edges_disk = []
    edges_half = []
    ax0, ay0 = anchor
    anchors_disk = []
    anchors_half = []
    anchors_gauss = []
    anchor_scale = []

    for M in mats:
        a, b, c, d = float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1])
        # Sides
        for side in sides:
            xs = side[:, 0].copy()
            ys = side[:, 1].copy()
            X2, Y2 = geo.apply_mobius_batch(a, b, c, d, xs, ys)
            edges_half.append(np.column_stack([X2, Y2]))
            U, V = geo.halfplane_to_disk_batch(X2, Y2)
            edges_disk.append(np.column_stack([U, V]))
        # Anchor
        ax, ay = geo.apply_mobius_batch(
            a, b, c, d, np.array([ax0]), np.array([ay0]))
        ux, vy = geo.halfplane_to_disk(ax[0], ay[0])
        anchors_disk.append([ux, vy])
        anchors_half.append([ax[0], ay[0]])
        mu, sig = geo.halfplane_to_gauss(ax[0], ay[0])
        anchors_gauss.append([mu, sig])
        anchor_scale.append(ay[0])

    edges_disk = np.array(edges_disk)
    edges_half = np.array(edges_half)
    anchors_disk = np.array(anchors_disk)
    anchors_half = np.array(anchors_half)
    anchors_gauss = np.array(anchors_gauss)
    anchor_scale = np.array(anchor_scale)
    return (edges_disk, edges_half, anchors_disk, anchors_half,
            anchors_gauss, anchor_scale, mats)


def halfplane_orbit_field(anchors_gauss, x_range=(-1.5, 1.5),
                          y_range=(0.06, 3.0), res=220):
    """
    Compute the minimum-hyperbolic-distance field to the tile anchors over a
    half-plane grid, for a diagnostic panel.

    Returns
    -------
    field : (res, res) array; gx, gy : the grid coordinates (half-plane).
    """
    xs = np.linspace(x_range[0], x_range[1], res)
    ys = np.linspace(y_range[0], y_range[1], res)
    gx, gy = np.meshgrid(xs, ys)
    ox = anchors_gauss[:, 0] / geo.SQRT2
    oy = anchors_gauss[:, 1]
    field = geo.orbit_distance_field(gx, gy, ox, oy)
    return field, gx, gy


# =============================================================================
# CASE 2: geodesic subdivision of the categorical octant
# =============================================================================

def _sphere_midpoint(a, b, radius):
    """Great-circle midpoint of two sphere points at fixed radius."""
    m = a + b
    m = m / np.linalg.norm(m) * radius
    return m


def subdivide_octant(depth=3, radius=2.0):
    """
    Geodesically subdivide the base categorical triangle (the positive octant of
    the radius-2 sphere) to a given depth.

    Returns
    -------
    triangles : (T, 3, 3) sphere-vertex triangles.
    edges : list of (2, 3) great-circle edge endpoints (deduplicated).
    """
    v0 = np.array([radius, 0.0, 0.0])
    v1 = np.array([0.0, radius, 0.0])
    v2 = np.array([0.0, 0.0, radius])
    tris = [(v0, v1, v2)]

    for _ in range(depth):
        new = []
        for (a, b, c) in tris:
            ab = _sphere_midpoint(a, b, radius)
            bc = _sphere_midpoint(b, c, radius)
            ca = _sphere_midpoint(c, a, radius)
            new.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        tris = new

    triangles = np.array([[t[0], t[1], t[2]] for t in tris])

    edge_set = {}
    edges = []
    for t in triangles:
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            key = tuple(np.round(np.concatenate([t[i], t[j]]), 6))
            rkey = tuple(np.round(np.concatenate([t[j], t[i]]), 6))
            if key not in edge_set and rkey not in edge_set:
                edge_set[key] = True
                edges.append(np.array([t[i], t[j]]))
    return triangles, edges


def octant_arcs(edges, n_samples=24):
    """Sample great-circle arcs for a list of edge endpoint pairs."""
    arcs = np.empty((len(edges), n_samples, 3), dtype=np.float64)
    for e, (p, q) in enumerate(edges):
        arcs[e] = geo.geodesic_arc_sphere(p, q, n_samples)
    return arcs


def sample_categorical(n=6, seed=7, k=3):
    """Sample categorical distributions from a symmetric Dirichlet prior."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(k), size=n)
    return P


def geodesic_circuits(n_points=28, n_frames=120, seed=11, radius=2.0):
    """
    Circulate a swarm of categorical distributions on closed geodesic circuits.

    A full great circle would leave the positive orthant and so leave the
    simplex. Each point therefore traverses a closed circuit of three
    great-circle arcs joining three sampled distributions, which stays inside
    the simplex by geodesic convexity of the orthant and is exactly periodic in
    the frame index. Phase offsets desynchronize the swarm.

    Returns
    -------
    positions : (n_frames, n_points, 3) sphere coordinates.
    entropies : (n_frames, n_points) Shannon entropy of each moving point.
    """
    rng = np.random.default_rng(seed)
    verts = []
    for _ in range(n_points):
        tri = rng.dirichlet(np.ones(3), size=3)
        verts.append([geo.simplex_to_sphere(tri[j]) for j in range(3)])

    positions = np.empty((n_frames, n_points, 3), dtype=np.float64)
    entropies = np.empty((n_frames, n_points), dtype=np.float64)
    phases = rng.random(n_points)

    for f in range(n_frames):
        base = f / n_frames
        for m in range(n_points):
            u = 3.0 * ((base + phases[m]) % 1.0)
            leg = int(u) % 3
            t = u - int(u)
            s = geo.slerp(verts[m][leg], verts[m][(leg + 1) % 3], t)
            positions[f, m] = s
            entropies[f, m] = geo.shannon_entropy(geo.sphere_to_simplex(s))
    return positions, entropies


# =============================================================================
# CASE 3: dually flat weave of the Gaussian family
# =============================================================================

def dual_grids(mu_range=(-2.0, 2.0), sigma_range=(0.4, 2.4),
               n_lines=9, n_samples=60):
    """
    Build the two conjugate coordinate grids of the Gaussian exponential family,
    both expressed in (mu, sigma) so their curvature contrast is visible.

    e-lines are straight in the natural coordinates theta; m-lines are straight
    in the expectation coordinates eta.

    Returns
    -------
    e_lines : (Ne, n_samples, 2), m_lines : (Nm, n_samples, 2).
    """
    mus = np.linspace(mu_range[0], mu_range[1], n_lines)
    sigs = np.linspace(sigma_range[0], sigma_range[1], n_lines)

    # Corner Gaussians define the parameter rectangle.
    corners = [(mu_range[0], sigma_range[0]), (mu_range[1], sigma_range[0]),
               (mu_range[1], sigma_range[1]), (mu_range[0], sigma_range[1])]
    thetas = np.array([geo.gauss_theta(m, s) for (m, s) in corners])
    etas = np.array([geo.gauss_eta(m, s) for (m, s) in corners])

    t1_lo, t1_hi = thetas[:, 0].min(), thetas[:, 0].max()
    t2_lo, t2_hi = thetas[:, 1].min(), thetas[:, 1].max()
    e1_lo, e1_hi = etas[:, 0].min(), etas[:, 0].max()
    e2_lo, e2_hi = etas[:, 1].min(), etas[:, 1].max()

    e_lines = []
    # theta1 = const, theta2 varies  and  theta2 = const, theta1 varies
    for t1 in np.linspace(t1_lo, t1_hi, n_lines):
        line = []
        for t2 in np.linspace(t2_lo, t2_hi, n_samples):
            if t2 < -1e-6:
                mu, sig = geo.theta_to_gauss(t1, t2)
                line.append([mu, sig])
        if len(line) > 1:
            e_lines.append(_resample(np.array(line), n_samples))
    for t2 in np.linspace(t2_lo, t2_hi, n_lines):
        if t2 >= -1e-6:
            continue
        line = []
        for t1 in np.linspace(t1_lo, t1_hi, n_samples):
            mu, sig = geo.theta_to_gauss(t1, t2)
            line.append([mu, sig])
        e_lines.append(_resample(np.array(line), n_samples))

    m_lines = []
    for e1 in np.linspace(e1_lo, e1_hi, n_lines):
        line = []
        for e2 in np.linspace(e2_lo, e2_hi, n_samples):
            if e2 - e1 * e1 > 1e-6:
                mu, sig = geo.eta_to_gauss(e1, e2)
                line.append([mu, sig])
        if len(line) > 1:
            m_lines.append(_resample(np.array(line), n_samples))
    for e2 in np.linspace(e2_lo, e2_hi, n_lines):
        line = []
        for e1 in np.linspace(e1_lo, e1_hi, n_samples):
            if e2 - e1 * e1 > 1e-6:
                mu, sig = geo.eta_to_gauss(e1, e2)
                line.append([mu, sig])
        if len(line) > 1:
            m_lines.append(_resample(np.array(line), n_samples))

    return np.array(e_lines), np.array(m_lines)


def _resample(line, n):
    """Resample a polyline to exactly n points by arc-length interpolation."""
    if len(line) == n:
        return line
    d = np.sqrt((np.diff(line, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-12:
        return np.repeat(line[:1], n, axis=0)
    s = s / s[-1]
    snew = np.linspace(0, 1, n)
    x = np.interp(snew, s, line[:, 0])
    y = np.interp(snew, s, line[:, 1])
    return np.column_stack([x, y])


def geodesic_mu_line(p, q, n_samples=60, kind='m'):
    """
    Polyline of the m- or e-geodesic between two Gaussians, in (mu, sigma).

    kind='m' interpolates linearly in expectation coordinates eta;
    kind='e' interpolates linearly in natural coordinates theta.
    """
    if kind == 'm':
        a = np.array(geo.gauss_eta(*p))
        b = np.array(geo.gauss_eta(*q))
        conv = geo.eta_to_gauss
    else:
        a = np.array(geo.gauss_theta(*p))
        b = np.array(geo.gauss_theta(*q))
        conv = geo.theta_to_gauss
    out = np.empty((n_samples, 2))
    for k in range(n_samples):
        t = k / (n_samples - 1.0)
        c = (1.0 - t) * a + t * b
        out[k] = conv(c[0], c[1])
    return out


def kl_asymmetry_grid(mu0, sigma0, mu_range=(-2.0, 2.0),
                      sigma_range=(0.4, 2.4), res=200):
    """Signed KL asymmetry field over a (mu, sigma) grid relative to p0."""
    mus = np.linspace(mu_range[0], mu_range[1], res)
    sigs = np.linspace(sigma_range[0], sigma_range[1], res)
    MU, SIG = np.meshgrid(mus, sigs)
    field = geo.kl_asymmetry_field(MU, SIG, mu0, sigma0)
    return field, MU, SIG


def dual_sweep_loop(Q_center, amp_mu, amp_sigma, n_frames):
    """
    Closed loop traversed by the moving reference distribution Q.

    The loop is an ellipse in (mu, sigma) centred on Q_center, traversed once
    per animation cycle so the sweep is exactly periodic. The sigma amplitude is
    clipped to keep the path strictly inside the positive half-plane.
    """
    mu_c, sigma_c = Q_center
    amp_sigma = min(amp_sigma, 0.75 * sigma_c)
    t = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    mu_q = mu_c + amp_mu * np.cos(t)
    sigma_q = sigma_c + amp_sigma * np.sin(t)
    return np.column_stack([mu_q, sigma_q])


def kl_asymmetry_sequence(Q_path, mu_range=(-2.0, 2.0),
                          sigma_range=(0.4, 2.4), res=140):
    """
    Divergence-asymmetry field for every frame of a sweep.

    The reference distribution is the moving Q, so the field itself evolves
    rather than serving as a static backdrop. This is the array that lets a
    downstream script reproduce the full spatiotemporal sequence from the
    archive alone.

    Returns
    -------
    fields : (F, res, res), MU, SIG : (res, res) grids.
    """
    mus = np.linspace(mu_range[0], mu_range[1], res)
    sigs = np.linspace(sigma_range[0], sigma_range[1], res)
    MU, SIG = np.meshgrid(mus, sigs)
    F = Q_path.shape[0]
    fields = np.empty((F, res, res), dtype=np.float64)
    for f in range(F):
        fields[f] = geo.kl_asymmetry_field(MU, SIG, Q_path[f, 0], Q_path[f, 1])
    return fields, MU, SIG


# =============================================================================
# CASE 4: hyperbolic Brownian motion of Gaussians
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _diffusion_kernel(x0, y0, n_frames, substeps, dt, seed):
    """
    Simulate hyperbolic Brownian motion in the standard half-plane chart.

    The driftless Ito system dx = y dW1, d(log y) = dW2 - (1/2) dt realizes half
    the Laplace-Beltrami operator of the standard half-plane. Integrating log y
    keeps y strictly positive by construction. Walkers are independent, so the
    ensemble is parallelized over the walker index.

    Returns
    -------
    X, Y : (n_frames, M) half-plane trajectories (frame 0 is the start).
    """
    M = x0.shape[0]
    X = np.empty((n_frames, M), dtype=np.float64)
    Y = np.empty((n_frames, M), dtype=np.float64)
    sqrt_dt = np.sqrt(dt)
    for m in prange(M):
        np.random.seed(seed + m)
        x = x0[m]
        ly = np.log(y0[m])
        X[0, m] = x
        Y[0, m] = np.exp(ly)
        for f in range(1, n_frames):
            for _ in range(substeps):
                y = np.exp(ly)
                z1 = np.random.normal(0.0, 1.0)
                z2 = np.random.normal(0.0, 1.0)
                x = x + y * sqrt_dt * z1
                ly = ly + sqrt_dt * z2 - 0.5 * dt
            X[f, m] = x
            Y[f, m] = np.exp(ly)
    return X, Y


def diffusion_simulate(mu0=0.0, sigma0=1.0, n_walkers=400, n_frames=120,
                       substeps=6, dt=0.01, seed=42):
    """
    Run the diffusion and return trajectories in Gaussian coordinates.

    Returns
    -------
    MU, SIG : (n_frames, M) Gaussian coordinates of the walkers.
    times : (n_frames,) metric diffusion times.
    """
    x0 = np.full(n_walkers, mu0 / geo.SQRT2)
    y0 = np.full(n_walkers, sigma0)
    X, Y = _diffusion_kernel(x0, y0, n_frames, substeps, dt, seed)
    MU = X * geo.SQRT2
    SIG = Y
    times = np.arange(n_frames) * substeps * dt
    return MU, SIG, times
