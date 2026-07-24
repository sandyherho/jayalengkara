"""
Unit tests for the FisherRaoModel engine.

The tests run each of the four cases at reduced resolution and check that the
result envelope is well formed, the frame coordinate is present, and the
headline correctness diagnostics land at the expected precision.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jayalengkara_fr.core.models import FisherRaoModel


@pytest.fixture(scope="module")
def model():
    return FisherRaoModel(n_cores=1, verbose=False)


def _common_envelope_ok(result):
    for key in ('case', 'kind', 'frame_kind', 'frames', 'metrics', 'params'):
        assert key in result
    assert np.asarray(result['frames']).ndim == 1
    assert len(result['frames']) > 0


class TestTessellation:

    def test_envelope(self, model):
        r = model.run({'case_type': 'gaussian_hyperbolic',
                       'max_tiles': 80, 'n_frames': 8, 'field_res': 40})
        _common_envelope_ok(r)
        assert r['kind'] == 'tessellation'

    def test_fundamental_domain_area(self, model):
        r = model.run({'case_type': 'gaussian_hyperbolic',
                       'max_tiles': 80, 'n_frames': 8, 'field_res': 40})
        assert np.isclose(r['metrics']['fundamental_domain_area'], np.pi / 3)

    def test_modular_relations_exact(self, model):
        r = model.run({'case_type': 'gaussian_hyperbolic',
                       'max_tiles': 80, 'n_frames': 8, 'field_res': 40})
        assert r['metrics']['modular_relations_residual'] == 0


class TestSphere:

    def test_envelope(self, model):
        r = model.run({'case_type': 'categorical_sphere',
                       'subdivision_depth': 2, 'n_frames': 8, 'field_res': 40})
        _common_envelope_ok(r)
        assert r['kind'] == 'sphere'

    def test_triangle_inequality(self, model):
        r = model.run({'case_type': 'categorical_sphere',
                       'subdivision_depth': 2, 'n_frames': 8, 'field_res': 40})
        assert r['metrics']['triangle_inequality_max_violation'] < 1e-9

    def test_entropy_uniform(self, model):
        r = model.run({'case_type': 'categorical_sphere',
                       'subdivision_depth': 2, 'n_frames': 8, 'field_res': 40})
        assert np.isclose(r['metrics']['entropy_uniform'], np.log(3.0))


class TestDualWeave:

    def test_envelope(self, model):
        r = model.run({'case_type': 'dual_weave',
                       'n_lines': 5, 'n_frames': 8, 'field_res': 40})
        _common_envelope_ok(r)
        assert r['kind'] == 'dual_weave'

    def test_pythagorean_residual(self, model):
        r = model.run({'case_type': 'dual_weave',
                       'n_lines': 5, 'n_frames': 8, 'field_res': 40})
        assert r['metrics']['pythagorean_residual'] < 1e-10


class TestDiffusion:

    def test_envelope(self, model):
        r = model.run({'case_type': 'hyperbolic_diffusion',
                       'n_walkers': 100, 'n_frames': 20, 'substeps': 4})
        _common_envelope_ok(r)
        assert r['kind'] == 'diffusion'

    def test_martingale_of_sigma(self, model):
        # E[sigma] is conserved; with a finite ensemble the deviation is small.
        r = model.run({'case_type': 'hyperbolic_diffusion',
                       'n_walkers': 800, 'n_frames': 30, 'substeps': 5})
        assert r['metrics']['martingale_deviation'] < 0.15

    def test_log_sigma_drifts_down(self, model):
        r = model.run({'case_type': 'hyperbolic_diffusion',
                       'n_walkers': 800, 'n_frames': 30, 'substeps': 5})
        assert r['metrics']['log_sigma_slope'] < 0.0

    def test_positive_escape_rate(self, model):
        r = model.run({'case_type': 'hyperbolic_diffusion',
                       'n_walkers': 400, 'n_frames': 30, 'substeps': 5})
        assert r['metrics']['escape_rate'] > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
