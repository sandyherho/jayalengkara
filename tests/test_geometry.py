"""
Unit tests for Fisher-Rao geometric primitives.

The tests verify the closed-form identities that make the four cases exact:
curvatures, the isometry of the square-root embedding, geodesic distances, the
generalized Pythagorean theorem, and the modular group relations.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jayalengkara_fr.core import geometry as geo
from jayalengkara_fr.core import diagnostics as diag


class TestGaussianHalfPlane:

    def test_chart_roundtrip(self):
        mu, sigma = 1.3, 0.7
        x, y = geo.gauss_to_halfplane(mu, sigma)
        mu2, sigma2 = geo.halfplane_to_gauss(x, y)
        assert np.isclose(mu, mu2)
        assert np.isclose(sigma, sigma2)

    def test_distance_symmetry(self):
        d1 = geo.fisher_rao_distance_gaussian(0.0, 1.0, 1.0, 2.0)
        d2 = geo.fisher_rao_distance_gaussian(1.0, 2.0, 0.0, 1.0)
        assert np.isclose(d1, d2)

    def test_distance_zero_on_diagonal(self):
        d = geo.fisher_rao_distance_gaussian(0.5, 1.2, 0.5, 1.2)
        assert np.isclose(d, 0.0, atol=1e-9)

    def test_curvature_is_minus_half(self):
        assert np.isclose(geo.K_GAUSSIAN, -0.5)
        assert diag.gaussian_curvature_check() < 1e-12

    def test_mobius_preserves_distance(self):
        # SL(2,R) acts by isometries of the half-plane.
        a, b, c, d = 2.0, 1.0, 1.0, 1.0  # det = 1
        X = np.array([0.0, 0.4])
        Y = np.array([1.0, 1.5])
        d0 = geo.hyperbolic_distance(X[0], Y[0], X[1], Y[1])
        Xo, Yo = geo.apply_mobius_batch(a, b, c, d, X, Y)
        d1 = geo.hyperbolic_distance(Xo[0], Yo[0], Xo[1], Yo[1])
        assert np.isclose(d0, d1, rtol=1e-9)


class TestCategoricalSphere:

    def test_embedding_radius(self):
        p = np.array([0.2, 0.3, 0.5])
        s = geo.simplex_to_sphere(p)
        assert np.isclose(np.linalg.norm(s), 2.0)

    def test_embedding_roundtrip(self):
        p = np.array([0.1, 0.6, 0.3])
        s = geo.simplex_to_sphere(p)
        p2 = geo.sphere_to_simplex(s)
        assert np.allclose(p, p2)

    def test_distance_matches_arc_length(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.2, 0.6])
        s1 = geo.simplex_to_sphere(p)
        s2 = geo.simplex_to_sphere(q)
        arc = geo.geodesic_arc_sphere(s1, s2, 4096)
        arc_len = np.sqrt(((arc[1:] - arc[:-1]) ** 2).sum(axis=1)).sum()
        closed = geo.fisher_rao_distance_categorical(p, q)
        assert np.isclose(arc_len, closed, atol=1e-6)

    def test_uniform_maximizes_entropy(self):
        uniform = np.array([1.0, 1.0, 1.0]) / 3.0
        skewed = np.array([0.8, 0.1, 0.1])
        assert geo.shannon_entropy(uniform) > geo.shannon_entropy(skewed)
        assert np.isclose(geo.shannon_entropy(uniform), np.log(3.0))

    def test_curvature_is_quarter(self):
        assert np.isclose(geo.K_CATEGORICAL, 0.25)


class TestExponentialFamily:

    def test_theta_roundtrip(self):
        mu, sigma = 0.7, 1.3
        t1, t2 = geo.gauss_theta(mu, sigma)
        mu2, sigma2 = geo.theta_to_gauss(t1, t2)
        assert np.isclose(mu, mu2)
        assert np.isclose(sigma, sigma2)

    def test_eta_roundtrip(self):
        mu, sigma = -0.4, 0.9
        e1, e2 = geo.gauss_eta(mu, sigma)
        mu2, sigma2 = geo.eta_to_gauss(e1, e2)
        assert np.isclose(mu, mu2)
        assert np.isclose(sigma, sigma2)

    def test_kl_nonnegative_and_zero_on_diagonal(self):
        assert np.isclose(geo.kl_gaussian(0.3, 1.1, 0.3, 1.1), 0.0, atol=1e-12)
        assert geo.kl_gaussian(0.0, 1.0, 1.0, 2.0) > 0.0

    def test_kl_asymmetry(self):
        d_pq = geo.kl_gaussian(0.0, 1.0, 1.0, 2.0)
        d_qp = geo.kl_gaussian(1.0, 2.0, 0.0, 1.0)
        assert not np.isclose(d_pq, d_qp)

    def test_generalized_pythagorean(self):
        triple = diag.pythagorean_triple(-1.0, 0.7, 0.6, 1.1)
        assert triple['pythagorean_residual'] < 1e-10
        assert np.isclose(triple['D_PR'], triple['D_PQ'] + triple['D_QR'], atol=1e-8)


class TestModularGroup:

    def test_relations_exact(self):
        assert diag.modular_relations_residual() == 0

    def test_generators_have_unit_determinant(self):
        from jayalengkara_fr.core.cases import modular_orbit
        mats = modular_orbit(60)
        for M in mats:
            assert np.linalg.det(M.astype(np.float64)) == pytest.approx(1.0, abs=1e-9)

    def test_orbit_is_deduplicated(self):
        from jayalengkara_fr.core.cases import modular_orbit, _psl_key
        mats = modular_orbit(120)
        keys = {_psl_key(M) for M in mats}
        assert len(keys) == len(mats)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
