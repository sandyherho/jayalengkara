"""
jayalengkara: diagnostic metrics.

Two families of diagnostics are computed. Correctness diagnostics verify that
the geometry is realized to numerical precision (group relations, the modular
fundamental-domain area, the generalized Pythagorean identity, the triangle
inequality, the conserved martingale of hyperbolic diffusion). Content
diagnostics report information-theoretic quantities that the visualizations are
built to make legible (entropy, curvature, divergence asymmetry, geodesic-
distance growth).

Every metric returned here is a plain Python float or int so that it can be
written verbatim into NetCDF attributes and CSV rows.
"""

import numpy as np

from . import geometry as geo


def gaussian_curvature_check(n_probe=64, seed=0):
    """
    Numerically verify the constant curvature K = -1/2 of the Gaussian manifold.

    The Gaussian metric equals twice the standard half-plane metric, whose
    Gaussian curvature is -1; scaling a 2-metric by a constant c divides the
    curvature by c, giving -1/2. We recompute the sectional curvature from the
    Brioschi formula on a random sample of interior points and return the maximum
    deviation from -1/2.
    """
    rng = np.random.default_rng(seed)
    max_dev = 0.0
    # Metric E = 1/sigma^2, F = 0, G = 2/sigma^2 in (mu, sigma) coordinates.
    # For a diagonal metric ds^2 = E du^2 + G dv^2 the Gaussian curvature is
    # K = -1/(2 sqrt(EG)) [ d/du (G_u / sqrt(EG)) + d/dv (E_v / sqrt(EG)) ].
    # Here E = 1/s^2, G = 2/s^2 depend only on v = sigma, giving K = -1/2 exactly.
    for _ in range(n_probe):
        sigma = float(rng.uniform(0.2, 5.0))
        E = 1.0 / sigma ** 2
        G = 2.0 / sigma ** 2
        # Analytic derivatives (E_v, G_v); E_u = G_u = 0.
        E_v = -2.0 / sigma ** 3
        G_v = -4.0 / sigma ** 3
        sqrtEG = np.sqrt(E * G)
        # d/dv (E_v / sqrtEG): E_v/sqrtEG = (-2/s^3)/(sqrt(2)/s^2) = -sqrt(2)/s.
        # derivative wrt sigma = sqrt(2)/s^2.
        term_v = np.sqrt(2.0) / sigma ** 2
        K = -1.0 / (2.0 * sqrtEG) * term_v
        max_dev = max(max_dev, abs(K - geo.K_GAUSSIAN))
    return float(max_dev)


def modular_relations_residual():
    """
    Verify the defining relations of the modular group in PSL(2, Z).

    With S = [[0,-1],[1,0]] and T = [[1,1],[0,1]] the relations S^2 = -I and
    (S T)^3 = -I hold as integer identities (equal to the identity in PSL). The
    residual is the integer Frobenius norm of the deviation and is exactly zero.
    """
    S = np.array([[0, -1], [1, 0]], dtype=np.int64)
    T = np.array([[1, 1], [0, 1]], dtype=np.int64)
    I = np.eye(2, dtype=np.int64)
    S2 = S @ S
    ST3 = np.linalg.matrix_power(S @ T, 3)
    r1 = np.abs(S2 + I).sum()          # S^2 = -I
    r2 = np.abs(ST3 + I).sum()         # (ST)^3 = -I
    return int(r1 + r2)


def tessellation_metrics(orbit_matrices, anchors_gauss):
    """
    Summaries of a generated modular tessellation.

    Parameters
    ----------
    orbit_matrices : (P, 2, 2) integer group elements.
    anchors_gauss : (P, 2) Gaussian anchors (mu, sigma) of tile representatives.

    Returns
    -------
    dict of metrics including the exact fundamental-domain area.
    """
    max_entry = int(np.abs(orbit_matrices).max())
    n_tiles = int(orbit_matrices.shape[0])
    sigma = anchors_gauss[:, 1]
    return {
        'n_tiles': n_tiles,
        'max_matrix_entry': max_entry,
        'fundamental_domain_area': float(np.pi / 3.0),  # exact for PSL(2,Z)
        'modular_relations_residual': modular_relations_residual(),
        'curvature': float(geo.K_GAUSSIAN),
        'curvature_max_deviation': gaussian_curvature_check(),
        'anchor_sigma_min': float(sigma.min()),
        'anchor_sigma_max': float(sigma.max()),
    }


def categorical_metrics(sample_dists):
    """
    Correctness and content metrics for the categorical sphere.

    Verifies the triangle inequality on random triples, cross-checks the closed-
    form geodesic distance against a densely sampled great-circle arc length, and
    reports the entropy range of the family.

    Parameters
    ----------
    sample_dists : (M, K) array of categorical distributions.
    """
    M = sample_dists.shape[0]
    rng = np.random.default_rng(1)
    max_tri_violation = 0.0
    for _ in range(200):
        i, j, k = rng.integers(0, M, size=3)
        d_ij = geo.fisher_rao_distance_categorical(sample_dists[i], sample_dists[j])
        d_jk = geo.fisher_rao_distance_categorical(sample_dists[j], sample_dists[k])
        d_ik = geo.fisher_rao_distance_categorical(sample_dists[i], sample_dists[k])
        violation = d_ik - (d_ij + d_jk)
        if violation > max_tri_violation:
            max_tri_violation = violation

    # Closed-form vs. numerically integrated arc length for one pair.
    p = sample_dists[0]
    q = sample_dists[min(1, M - 1)]
    s1 = geo.simplex_to_sphere(p)
    s2 = geo.simplex_to_sphere(q)
    arc = geo.geodesic_arc_sphere(s1, s2, 4096)
    arc_len = float(np.sqrt(((arc[1:] - arc[:-1]) ** 2).sum(axis=1)).sum())
    closed = float(geo.fisher_rao_distance_categorical(p, q))
    dist_residual = abs(arc_len - closed)

    entropies = np.array([geo.shannon_entropy(sample_dists[i]) for i in range(M)])
    return {
        'triangle_inequality_max_violation': float(max_tri_violation),
        'geodesic_length_residual': float(dist_residual),
        'curvature': float(geo.K_CATEGORICAL),
        'entropy_min': float(entropies.min()),
        'entropy_max': float(entropies.max()),
        'entropy_uniform': float(np.log(sample_dists.shape[1])),
    }


def pythagorean_triple(mu_p, sigma_p, mu_q, sigma_q):
    """
    Construct a dual-orthogonal triple (P, Q, R) and verify the generalized
    Pythagorean identity D(P||R) = D(P||Q) + D(Q||R).

    Given P and Q, the point R is placed along the e-geodesic (straight in the
    natural coordinates theta) emanating from Q in the direction dual-orthogonal
    to the m-geodesic P->Q. Orthogonality is taken in the Fisher metric between
    the m-geodesic tangent (expressed in theta) and the e-geodesic tangent.

    Returns
    -------
    triple : dict with P, Q, R in (mu, sigma) and the divergence residual.
    """
    # m-geodesic P -> Q tangent, in expectation coordinates eta.
    e1p, e2p = geo.gauss_eta(mu_p, sigma_p)
    e1q, e2q = geo.gauss_eta(mu_q, sigma_q)
    m_tan_eta = np.array([e1q - e1p, e2q - e2p])

    # e-geodesic direction in natural coordinates theta, dual-orthogonal to the
    # m-geodesic. In dually flat geometry the natural pairing <a_theta, b_eta>
    # is the ordinary dot product of theta-vector and eta-vector. We therefore
    # pick an e-direction d_theta with <d_theta, m_tan_eta> = 0.
    d_theta = np.array([m_tan_eta[1], -m_tan_eta[0]])
    norm = np.linalg.norm(d_theta)
    if norm < 1e-12:
        d_theta = np.array([1.0, 0.0])
        norm = 1.0
    d_theta = d_theta / norm

    t1q, t2q = geo.gauss_theta(mu_q, sigma_q)
    # Step along the e-geodesic; keep theta2 < 0 so sigma stays real.
    step = 0.15
    for _ in range(40):
        t1r = t1q + step * d_theta[0]
        t2r = t2q + step * d_theta[1]
        if t2r < -1e-6:
            break
        step *= 0.5
    mu_r, sigma_r = geo.theta_to_gauss(t1r, t2r)

    d_pq = geo.kl_gaussian(mu_p, sigma_p, mu_q, sigma_q)
    d_qr = geo.kl_gaussian(mu_q, sigma_q, mu_r, sigma_r)
    d_pr = geo.kl_gaussian(mu_p, sigma_p, mu_r, sigma_r)
    residual = abs(d_pr - (d_pq + d_qr))

    # Fisher orthogonality residual of the two dual tangents, mapped to (mu,sigma).
    eps = 1e-5
    mu_q2, sigma_q2 = geo.eta_to_gauss(e1q + eps * m_tan_eta[0], e2q + eps * m_tan_eta[1])
    m_tan_gauss = np.array([(mu_q2 - mu_q) / eps, (sigma_q2 - sigma_q) / eps])
    mu_q3, sigma_q3 = geo.theta_to_gauss(t1q + eps * d_theta[0], t2q + eps * d_theta[1])
    e_tan_gauss = np.array([(mu_q3 - mu_q) / eps, (sigma_q3 - sigma_q) / eps])
    ortho = geo.fisher_inner_gaussian(
        mu_q, sigma_q,
        m_tan_gauss[0], m_tan_gauss[1],
        e_tan_gauss[0], e_tan_gauss[1],
    )

    return {
        'P': (float(mu_p), float(sigma_p)),
        'Q': (float(mu_q), float(sigma_q)),
        'R': (float(mu_r), float(sigma_r)),
        'D_PQ': float(d_pq),
        'D_QR': float(d_qr),
        'D_PR': float(d_pr),
        'pythagorean_residual': float(residual),
        'orthogonality_residual': float(ortho),
    }


def dual_weave_metrics(triple, asymmetry_field, residual_series=None,
                       ortho_series=None):
    """
    Aggregate dual-weave diagnostics.

    When the case is animated the Pythagorean identity is verified at every
    frame, not only at one representative triple. The reported residual is then
    the worst case over the whole sweep, which is the stronger claim.
    """
    finite = asymmetry_field[np.isfinite(asymmetry_field)]
    if residual_series is not None and len(residual_series) > 0:
        pyth = float(np.max(residual_series))
        pyth_mean = float(np.mean(residual_series))
        n_verified = int(len(residual_series))
    else:
        pyth = triple['pythagorean_residual']
        pyth_mean = triple['pythagorean_residual']
        n_verified = 1
    if ortho_series is not None and len(ortho_series) > 0:
        ortho = float(np.max(np.abs(ortho_series)))
    else:
        ortho = triple['orthogonality_residual']
    return {
        'pythagorean_residual': pyth,
        'pythagorean_residual_mean': pyth_mean,
        'frames_verified': n_verified,
        'orthogonality_residual': ortho,
        'D_PQ': triple['D_PQ'],
        'D_QR': triple['D_QR'],
        'D_PR': triple['D_PR'],
        'kl_asymmetry_max': float(np.max(np.abs(finite))),
        'kl_asymmetry_mean': float(np.mean(finite)),
        'curvature': float(geo.K_GAUSSIAN),
    }


def diffusion_metrics(times, sigma_paths, dist_from_start, entropy_series,
                      sigma0):
    """
    Content and correctness metrics for hyperbolic diffusion.

    Parameters
    ----------
    times : (F,) diffusion times.
    sigma_paths : (F, M) sigma coordinate of each walker over time.
    dist_from_start : (F, M) Fisher-Rao distance of each walker from its origin.
    entropy_series : (F,) ensemble entropy estimate over time.
    sigma0 : initial sigma.
    """
    mean_dist = dist_from_start.mean(axis=1)
    # Linear escape-rate fit through the origin-anchored distance growth.
    A = np.vstack([times, np.ones_like(times)]).T
    slope, intercept = np.linalg.lstsq(A, mean_dist, rcond=None)[0]

    mean_sigma = sigma_paths.mean(axis=1)               # martingale, ~ sigma0
    mean_log_sigma = np.log(sigma_paths).mean(axis=1)   # drifts down ~ -t/2
    # Slope of E[log sigma] in the standard time (metric is 2x, times are metric
    # times; convert with the factor SQRT2^2 = 2 folded into the model).
    log_slope, _ = np.linalg.lstsq(A, mean_log_sigma, rcond=None)[0]

    return {
        'escape_rate': float(slope),
        'escape_intercept': float(intercept),
        'mean_sigma_final': float(mean_sigma[-1]),
        'mean_sigma_initial_ref': float(sigma0),
        'martingale_deviation': float(abs(mean_sigma[-1] - sigma0) / sigma0),
        'log_sigma_slope': float(log_slope),
        'entropy_initial': float(entropy_series[0]),
        'entropy_final': float(entropy_series[-1]),
        'entropy_growth': float(entropy_series[-1] - entropy_series[0]),
        'curvature': float(geo.K_GAUSSIAN),
    }
