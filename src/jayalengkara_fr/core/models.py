"""
jayalengkara: the Fisher-Rao model engine.

The engine dispatches a configuration to one of the four case builders and
returns a single result dictionary with a common envelope. Every builder writes
raw geometry, a frame coordinate (time, zoom angle, or sweep parameter), the
diagnostic metrics, and the reproducibility parameters. The visualization and
I/O layers consume this envelope without knowing the case internals.
"""

import numpy as np

from . import geometry as geo
from . import cases
from . import diagnostics as diag


CASE_KINDS = {
    'gaussian_hyperbolic': 'tessellation',
    'categorical_sphere': 'sphere',
    'dual_weave': 'dual_weave',
    'hyperbolic_diffusion': 'diffusion',
}


class FisherRaoModel:
    """Engine that realizes a Fisher-Rao case as renderable, archivable geometry."""

    def __init__(self, n_cores=None, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger
        self.n_cores = geo.configure_threads(n_cores)
        if verbose:
            print(f"  CPU cores: {self.n_cores}")
            print(f"  Numba: {'ENABLED' if geo.NUMBA_AVAILABLE else 'DISABLED'}")

    def run(self, config):
        """Dispatch a validated configuration to the appropriate builder."""
        case = config.get('case_type', 'gaussian_hyperbolic')
        if case == 'gaussian_hyperbolic':
            return self._run_tessellation(config)
        if case == 'categorical_sphere':
            return self._run_sphere(config)
        if case == 'dual_weave':
            return self._run_dual_weave(config)
        if case == 'hyperbolic_diffusion':
            return self._run_diffusion(config)
        raise ValueError(f"Unknown case_type: {case}")

    # ------------------------------------------------------------------ case 1
    def _run_tessellation(self, config):
        if self.verbose:
            print("    Generating modular tessellation (exact integer orbit)...")
        max_tiles = int(config.get('max_tiles', 160))
        n_samples = int(config.get('edge_samples', 48))
        n_frames = int(config.get('n_frames', 120))
        res = int(config.get('field_res', 220))

        (edges, edges_half, anchors_disk, anchors_half,
         anchors_gauss, scale, mats) = cases.tessellation_geometry(
            max_tiles=max_tiles, n_samples=n_samples)
        field, gx, gy = cases.halfplane_orbit_field(anchors_gauss, res=res)

        # The frame coordinate is the parameter of the parabolic isometry
        # z -> z + s. Because T: z -> z + 1 belongs to the modular group, the
        # tessellation maps exactly onto itself at s = 1, so the flow is both a
        # genuine hyperbolic isometry and exactly periodic over one cycle.
        frames = np.linspace(0.0, 1.0, n_frames, endpoint=False)
        metrics = diag.tessellation_metrics(mats, anchors_gauss)

        if self.verbose:
            print(f"      Tiles: {metrics['n_tiles']}  "
                  f"max |entry|: {metrics['max_matrix_entry']}  "
                  f"fund. area: {metrics['fundamental_domain_area']:.6f}")

        return {
            'case': config.get('scenario_name', 'gaussian_hyperbolic'),
            'kind': 'tessellation',
            'frame_kind': 'parabolic_flow',
            'frames': frames,
            'edges_disk': edges,
            'edges_half': edges_half,
            'anchors_disk': anchors_disk,
            'anchors_half': anchors_half,
            'anchors_gauss': anchors_gauss,
            'anchor_scale': scale,
            'field': field,
            'field_x': gx,
            'field_y': gy,
            'metrics': metrics,
            'params': {
                'max_tiles': max_tiles, 'edge_samples': n_samples,
                'field_res': res, 'n_frames': n_frames,
                'n_cores': self.n_cores, 'numba_enabled': int(geo.NUMBA_AVAILABLE),
                'curvature': geo.K_GAUSSIAN,
            },
        }

    # ------------------------------------------------------------------ case 2
    def _run_sphere(self, config):
        if self.verbose:
            print("    Subdividing categorical octant (geodesic refinement)...")
        depth = int(config.get('subdivision_depth', 3))
        n_samples = int(config.get('edge_samples', 24))
        n_frames = int(config.get('n_frames', 120))
        n_geo = int(config.get('n_sample_geodesics', 4))
        res = int(config.get('field_res', 200))

        triangles, edges = cases.subdivide_octant(depth=depth)
        arcs = cases.octant_arcs(edges, n_samples=n_samples)

        centroids = triangles.mean(axis=1)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True) * 2.0
        centroid_p = np.array([geo.sphere_to_simplex(c) for c in centroids])

        dists = cases.sample_categorical(n=max(3, n_geo + 1), seed=7)
        geodesics = []
        for i in range(n_geo):
            s1 = geo.simplex_to_sphere(dists[i])
            s2 = geo.simplex_to_sphere(dists[i + 1])
            geodesics.append(geo.geodesic_arc_sphere(s1, s2, 48))
        geodesics = np.array(geodesics)

        # Tracer walks a closed geodesic triangle between three distributions.
        tri_pts = [geo.simplex_to_sphere(dists[j]) for j in range(3)]
        tracer = np.empty((n_frames, 3))
        for f in range(n_frames):
            u = 3.0 * f / n_frames
            leg = int(u) % 3
            t = u - int(u)
            tracer[f] = geo.slerp(tri_pts[leg], tri_pts[(leg + 1) % 3], t)

        # A swarm of distributions circulating on closed geodesic circuits, so
        # the manifold is populated by moving probability measures rather than a
        # static mesh viewed from a turning camera.
        n_orbit = int(config.get('n_orbit_points', 28))
        orbit_points, orbit_entropy = cases.geodesic_circuits(
            n_points=n_orbit, n_frames=n_frames,
            seed=int(config.get('seed', 42)))

        # Entropy field over the barycentric grid.
        aa = np.linspace(0.0, 1.0, res)
        bb = np.linspace(0.0, 1.0, res)
        A, B = np.meshgrid(aa, bb)
        efield = geo.entropy_field_ternary(A, B)

        frames = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
        metrics = diag.categorical_metrics(centroid_p)

        if self.verbose:
            print(f"      Triangles: {triangles.shape[0]}  "
                  f"tri-ineq viol: {metrics['triangle_inequality_max_violation']:.2e}  "
                  f"geo residual: {metrics['geodesic_length_residual']:.2e}")

        return {
            'case': config.get('scenario_name', 'categorical_sphere'),
            'kind': 'sphere',
            'frame_kind': 'azimuth',
            'frames': frames,
            'arcs': arcs,
            'centroids': centroids,
            'centroid_p': centroid_p,
            'geodesics': geodesics,
            'tracer': tracer,
            'orbit_points': orbit_points,
            'orbit_entropy': orbit_entropy,
            'entropy_field': efield,
            'field_a': A,
            'field_b': B,
            'metrics': metrics,
            'params': {
                'subdivision_depth': depth, 'edge_samples': n_samples,
                'n_sample_geodesics': n_geo, 'n_orbit_points': n_orbit,
                'field_res': res,
                'n_frames': n_frames, 'n_cores': self.n_cores,
                'numba_enabled': int(geo.NUMBA_AVAILABLE),
                'curvature': geo.K_CATEGORICAL,
            },
        }

    # ------------------------------------------------------------------ case 3
    def _run_dual_weave(self, config):
        if self.verbose:
            print("    Building dually flat weave (theta / eta grids)...")
        n_lines = int(config.get('n_lines', 9))
        n_samples = int(config.get('line_samples', 60))
        n_frames = int(config.get('n_frames', 120))
        res = int(config.get('field_res', 200))

        e_lines, m_lines = cases.dual_grids(n_lines=n_lines, n_samples=n_samples)

        P = (float(config.get('P_mu', -1.0)), float(config.get('P_sigma', 0.7)))
        Q_center = (float(config.get('Q_mu', 0.6)), float(config.get('Q_sigma', 1.1)))
        amp_mu = float(config.get('loop_mu_amp', 0.9))
        amp_sigma = float(config.get('loop_sigma_amp', 0.35))

        # Q travels a closed loop, so the dual-orthogonal triple, both geodesic
        # legs, and the divergence field all evolve together over the cycle. The
        # Pythagorean identity is re-verified at every frame.
        Q_path = cases.dual_sweep_loop(Q_center, amp_mu, amp_sigma, n_frames)

        triples = np.empty((n_frames, 3, 2))
        m_legs = np.empty((n_frames, n_samples, 2))
        e_legs = np.empty((n_frames, n_samples, 2))
        residuals = np.empty(n_frames)
        orthos = np.empty(n_frames)
        divergences = np.empty((n_frames, 3))

        for f in range(n_frames):
            Qf = (Q_path[f, 0], Q_path[f, 1])
            tri = diag.pythagorean_triple(P[0], P[1], Qf[0], Qf[1])
            Rf = tri['R']
            triples[f] = np.vstack([tri['P'], tri['Q'], tri['R']])
            m_legs[f] = cases.geodesic_mu_line(P, Qf, n_samples, kind='m')
            e_legs[f] = cases.geodesic_mu_line(Qf, Rf, n_samples, kind='e')
            residuals[f] = tri['pythagorean_residual']
            orthos[f] = tri['orthogonality_residual']
            divergences[f] = [tri['D_PQ'], tri['D_QR'], tri['D_PR']]
            if f == 0:
                triple0 = tri

        fields, MU, SIG = cases.kl_asymmetry_sequence(Q_path, res=res)

        frames = np.linspace(0.0, 1.0, n_frames, endpoint=False)
        metrics = diag.dual_weave_metrics(triple0, fields[0], residuals, orthos)

        if self.verbose:
            print(f"      Pythagorean residual (worst of {n_frames} frames): "
                  f"{metrics['pythagorean_residual']:.2e}")

        return {
            'case': config.get('scenario_name', 'dual_weave'),
            'kind': 'dual_weave',
            'frame_kind': 'sweep',
            'frames': frames,
            'e_lines': e_lines,
            'm_lines': m_lines,
            'triples': triples,
            'Q_path': Q_path,
            'm_legs': m_legs,
            'e_legs': e_legs,
            'residual_series': residuals,
            'divergence_series': divergences,
            'asymmetry_fields': fields,
            'field_mu': MU,
            'field_sigma': SIG,
            'metrics': metrics,
            'params': {
                'n_lines': n_lines, 'line_samples': n_samples,
                'field_res': res, 'n_frames': n_frames,
                'P_mu': P[0], 'P_sigma': P[1],
                'Q_mu': Q_center[0], 'Q_sigma': Q_center[1],
                'loop_mu_amp': amp_mu, 'loop_sigma_amp': amp_sigma,
                'n_cores': self.n_cores,
                'numba_enabled': int(geo.NUMBA_AVAILABLE),
                'curvature': geo.K_GAUSSIAN,
            },
        }

    # ------------------------------------------------------------------ case 4
    def _run_diffusion(self, config):
        if self.verbose:
            print("    Simulating hyperbolic Brownian motion of Gaussians...")
        mu0 = float(config.get('mu0', 0.0))
        sigma0 = float(config.get('sigma0', 1.0))
        n_walkers = int(config.get('n_walkers', 400))
        n_frames = int(config.get('n_frames', 120))
        substeps = int(config.get('substeps', 6))
        dt = float(config.get('dt', 0.01))
        seed = int(config.get('seed', 42))

        MU, SIG, times = cases.diffusion_simulate(
            mu0=mu0, sigma0=sigma0, n_walkers=n_walkers, n_frames=n_frames,
            substeps=substeps, dt=dt, seed=seed)

        # Disk coordinates for rendering.
        X = MU / geo.SQRT2
        UD = np.empty_like(X)
        VD = np.empty_like(X)
        for f in range(n_frames):
            u, v = geo.halfplane_to_disk_batch(X[f], SIG[f])
            UD[f] = u
            VD[f] = v

        # Fisher-Rao distance of each walker from its start.
        dist = np.empty((n_frames, n_walkers))
        for f in range(n_frames):
            for m in range(n_walkers):
                dist[f, m] = geo.fisher_rao_distance_gaussian(
                    mu0, sigma0, MU[f, m], SIG[f, m])

        # Ensemble entropy estimate in the flat (x, log y) chart via a histogram.
        entropy_series = np.empty(n_frames)
        for f in range(n_frames):
            xf = X[f]
            lyf = np.log(SIG[f])
            H, xe, ye = np.histogram2d(xf, lyf, bins=24)
            p = H / H.sum()
            p = p[p > 0]
            dx = (xe[1] - xe[0]) * (ye[1] - ye[0])
            entropy_series[f] = -np.sum(p * np.log(p)) + np.log(dx)

        # Reference geodesic: the vertical geodesic from the start point.
        ref_x, ref_y = geo.halfplane_geodesic(
            mu0 / geo.SQRT2, sigma0, mu0 / geo.SQRT2, sigma0 * 6.0, 60)
        ref_u, ref_v = geo.halfplane_to_disk_batch(ref_x, ref_y)
        ref_disk = np.column_stack([ref_u, ref_v])

        metrics = diag.diffusion_metrics(times, SIG, dist, entropy_series, sigma0)

        if self.verbose:
            print(f"      Walkers: {n_walkers}  escape rate: {metrics['escape_rate']:.4f}  "
                  f"martingale dev: {metrics['martingale_deviation']:.2e}")

        return {
            'case': config.get('scenario_name', 'hyperbolic_diffusion'),
            'kind': 'diffusion',
            'frame_kind': 'time',
            'frames': times,
            'mu': MU,
            'sigma': SIG,
            'disk_u': UD,
            'disk_v': VD,
            'dist_from_start': dist,
            'entropy_series': entropy_series,
            'ref_geodesic_disk': ref_disk,
            'metrics': metrics,
            'params': {
                'mu0': mu0, 'sigma0': sigma0, 'n_walkers': n_walkers,
                'n_frames': n_frames, 'substeps': substeps, 'dt': dt,
                'seed': seed, 'n_cores': self.n_cores,
                'numba_enabled': int(geo.NUMBA_AVAILABLE),
                'curvature': geo.K_GAUSSIAN,
            },
        }
