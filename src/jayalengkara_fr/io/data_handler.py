"""
Data handler for Fisher-Rao results.

NetCDF is the archival source of truth. Each case writes its raw geometry, its
frame coordinate, and every parameter needed to reproduce the spatiotemporal
figures, so that a downstream script can regenerate any plot from the .nc file
alone. Diagnostic metrics and reproducibility parameters are stored as CF-style
global attributes. The CSV writers provide flat metric tables, including a
cross-case comparison table.
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

_VERSION = "0.0.1"


def _put_attrs(nc, prefix, d):
    """Store a flat dict of scalars as prefixed global attributes."""
    for k, v in d.items():
        if isinstance(v, (tuple, list, np.ndarray)):
            v = np.asarray(v, dtype=np.float64).ravel()
            nc.setncattr(f"{prefix}{k}", v)
        elif isinstance(v, bool):
            nc.setncattr(f"{prefix}{k}", int(v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            nc.setncattr(f"{prefix}{k}", float(v))
        else:
            nc.setncattr(f"{prefix}{k}", str(v))


def _var(nc, name, data, dims, long_name, desc, units="dimensionless",
         dtype='f4'):
    """
    Create a compressed float variable and attach metadata.

    Geometry is stored at single precision, which is ample for rendering. Series
    that carry a machine-precision claim are stored at double precision with
    dtype='f8', so that the identity can be re-verified from the archive alone
    rather than being limited by the storage format.
    """
    v = nc.createVariable(name, dtype, dims, zlib=True, complevel=4)
    v[:] = data
    v.units = units
    v.long_name = long_name
    v.description = desc
    return v


class DataHandler:
    """NetCDF and CSV output for Fisher-Rao cases."""

    @staticmethod
    def save_netcdf(filename, result, metadata, output_dir="outputs"):
        """Dispatch NetCDF writing by case kind."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename

        kind = result['kind']
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            frames = np.asarray(result['frames'], dtype=np.float64)
            nc.createDimension('frame', frames.shape[0])
            fv = nc.createVariable('frame', 'f4', ('frame',), zlib=True)
            fv[:] = frames
            fv.long_name = 'frame_coordinate'
            fv.frame_kind = result['frame_kind']
            fv.description = ("Animation frame coordinate: time, rotation angle, "
                              "azimuth, or sweep parameter depending on the case")

            if kind == 'tessellation':
                DataHandler._write_tessellation(nc, result)
            elif kind == 'sphere':
                DataHandler._write_sphere(nc, result)
            elif kind == 'dual_weave':
                DataHandler._write_dual_weave(nc, result)
            elif kind == 'diffusion':
                DataHandler._write_diffusion(nc, result)

            _put_attrs(nc, 'metric_', result['metrics'])
            _put_attrs(nc, 'param_', result['params'])

            nc.case = str(result['case'])
            nc.kind = kind
            nc.frame_kind = result['frame_kind']
            nc.scenario = metadata.get('scenario_name', 'unknown')
            nc.case_type = metadata.get('case_type', 'unknown')
            nc.created = datetime.now().isoformat()
            nc.software = "jayalengkara"
            nc.version = _VERSION
            nc.method = "fisher_rao_information_geometry"
            nc.Conventions = "CF-1.8"
            nc.title = f"Fisher-Rao: {metadata.get('scenario_name', 'unknown')}"
            nc.institution = "Applied Geology Research Group ITB"
            nc.license = "MIT"
            nc.history = f"Created {datetime.now().isoformat()}"
        return str(filepath)

    # ---------------------------------------------------------------- writers
    @staticmethod
    def _write_tessellation(nc, r):
        edges = r['edges_disk']
        nc.createDimension('edge', edges.shape[0])
        nc.createDimension('edge_sample', edges.shape[1])
        nc.createDimension('plane', 2)
        nc.createDimension('anchor', r['anchors_disk'].shape[0])
        nc.createDimension('grid_y', r['field'].shape[0])
        nc.createDimension('grid_x', r['field'].shape[1])

        _var(nc, 'edges_disk', edges, ('edge', 'edge_sample', 'plane'),
             'tile_sides_disk', 'Modular tile side polylines in Poincare disk coordinates')
        _var(nc, 'edges_half', r['edges_half'], ('edge', 'edge_sample', 'plane'),
             'tile_sides_halfplane',
             'Tile sides in half-plane coordinates; apply z -> z + frame then the '
             'Cayley transform to replay the parabolic isometry flow')
        _var(nc, 'anchors_disk', r['anchors_disk'], ('anchor', 'plane'),
             'tile_anchors_disk', 'Tile representative points in disk coordinates')
        _var(nc, 'anchors_half', r['anchors_half'], ('anchor', 'plane'),
             'tile_anchors_halfplane', 'Tile representatives in half-plane coordinates')
        _var(nc, 'anchors_gauss', r['anchors_gauss'], ('anchor', 'plane'),
             'tile_anchors_gauss', 'Tile representatives in Gaussian (mu, sigma)')
        _var(nc, 'anchor_scale', r['anchor_scale'], ('anchor',),
             'anchor_scale', 'Local hyperbolic scale (half-plane height) per anchor')
        _var(nc, 'orbit_distance_field', r['field'], ('grid_y', 'grid_x'),
             'orbit_distance_field', 'Minimum hyperbolic distance to tile anchors')
        _var(nc, 'field_x', r['field_x'], ('grid_y', 'grid_x'),
             'field_x', 'Half-plane x coordinate of the field grid')
        _var(nc, 'field_y', r['field_y'], ('grid_y', 'grid_x'),
             'field_y', 'Half-plane y (sigma) coordinate of the field grid')

    @staticmethod
    def _write_sphere(nc, r):
        arcs = r['arcs']
        nc.createDimension('arc', arcs.shape[0])
        nc.createDimension('arc_sample', arcs.shape[1])
        nc.createDimension('space', 3)
        nc.createDimension('centroid', r['centroids'].shape[0])
        nc.createDimension('outcome', r['centroid_p'].shape[1])
        nc.createDimension('geodesic', r['geodesics'].shape[0])
        nc.createDimension('geo_sample', r['geodesics'].shape[1])
        nc.createDimension('grid_b', r['entropy_field'].shape[0])
        nc.createDimension('grid_a', r['entropy_field'].shape[1])

        _var(nc, 'arcs', arcs, ('arc', 'arc_sample', 'space'),
             'octant_edges', 'Great-circle tile edges on the radius-2 sphere')
        _var(nc, 'centroids', r['centroids'], ('centroid', 'space'),
             'triangle_centroids', 'Sub-triangle centroids on the sphere')
        _var(nc, 'centroid_p', r['centroid_p'], ('centroid', 'outcome'),
             'centroid_distributions', 'Categorical distributions at centroids')
        _var(nc, 'geodesics', r['geodesics'], ('geodesic', 'geo_sample', 'space'),
             'sample_geodesics', 'Great-circle geodesics between sample distributions')
        _var(nc, 'tracer', r['tracer'], ('frame', 'space'),
             'geodesic_tracer', 'Point tracing a closed geodesic triangle over frames')
        nc.createDimension('orbit_point', r['orbit_points'].shape[1])
        _var(nc, 'orbit_points', r['orbit_points'], ('frame', 'orbit_point', 'space'),
             'circulating_distributions',
             'Swarm of categorical distributions circulating on closed geodesic circuits')
        _var(nc, 'orbit_entropy', r['orbit_entropy'], ('frame', 'orbit_point'),
             'circulating_entropy', 'Shannon entropy of each circulating distribution')
        _var(nc, 'entropy_field', r['entropy_field'], ('grid_b', 'grid_a'),
             'entropy_field', 'Shannon entropy over the barycentric grid')
        _var(nc, 'field_a', r['field_a'], ('grid_b', 'grid_a'),
             'field_a', 'Barycentric coordinate p0 of the field grid')
        _var(nc, 'field_b', r['field_b'], ('grid_b', 'grid_a'),
             'field_b', 'Barycentric coordinate p1 of the field grid')

    @staticmethod
    def _write_dual_weave(nc, r):
        e = r['e_lines']
        m = r['m_lines']
        nc.createDimension('e_line', e.shape[0])
        nc.createDimension('m_line', m.shape[0])
        nc.createDimension('line_sample', e.shape[1])
        nc.createDimension('plane', 2)
        nc.createDimension('leg_sample', r['m_legs'].shape[1])
        nc.createDimension('grid_s', r['asymmetry_fields'].shape[1])
        nc.createDimension('grid_m', r['asymmetry_fields'].shape[2])
        nc.createDimension('triple', 3)

        _var(nc, 'e_lines', e, ('e_line', 'line_sample', 'plane'),
             'e_geodesics', 'Natural-coordinate (e) geodesic grid in (mu, sigma)')
        _var(nc, 'm_lines', m, ('m_line', 'line_sample', 'plane'),
             'm_geodesics', 'Expectation-coordinate (m) geodesic grid in (mu, sigma)')
        _var(nc, 'm_legs', r['m_legs'], ('frame', 'leg_sample', 'plane'),
             'pythagorean_m_leg', 'm-geodesic P to Q of the triple, per frame')
        _var(nc, 'e_legs', r['e_legs'], ('frame', 'leg_sample', 'plane'),
             'pythagorean_e_leg', 'e-geodesic Q to R of the triple, per frame')
        _var(nc, 'pythagorean_triples', r['triples'], ('frame', 'triple', 'plane'),
             'pythagorean_triple', 'Points P, Q, R in (mu, sigma), per frame')
        _var(nc, 'Q_path', r['Q_path'], ('frame', 'plane'),
             'sweep_path', 'Closed loop traversed by the moving reference Q')
        _var(nc, 'residual_series', r['residual_series'], ('frame',),
             'pythagorean_residual_series',
             'Generalized Pythagorean residual verified at each frame', dtype='f8')
        _var(nc, 'divergence_series', r['divergence_series'], ('frame', 'triple'),
             'divergence_series', 'D(P||Q), D(Q||R), D(P||R) at each frame',
             dtype='f8')
        _var(nc, 'pythagorean_triples_f8', r['triples'], ('frame', 'triple', 'plane'),
             'pythagorean_triple_exact',
             'Points P, Q, R at double precision for exact re-verification',
             dtype='f8')
        _var(nc, 'asymmetry_fields', r['asymmetry_fields'],
             ('frame', 'grid_s', 'grid_m'), 'kl_asymmetry_field',
             'Signed KL asymmetry D(p||Q) - D(Q||p) relative to the moving Q')
        _var(nc, 'field_mu', r['field_mu'], ('grid_s', 'grid_m'),
             'field_mu', 'mu coordinate of the field grid')
        _var(nc, 'field_sigma', r['field_sigma'], ('grid_s', 'grid_m'),
             'field_sigma', 'sigma coordinate of the field grid')

    @staticmethod
    def _write_diffusion(nc, r):
        MU = r['mu']
        nc.createDimension('walker', MU.shape[1])
        nc.createDimension('plane', 2)
        nc.createDimension('ref_sample', r['ref_geodesic_disk'].shape[0])

        _var(nc, 'mu', MU, ('frame', 'walker'),
             'walker_mu', 'Walker mean coordinate over time')
        _var(nc, 'sigma', r['sigma'], ('frame', 'walker'),
             'walker_sigma', 'Walker standard-deviation coordinate over time')
        _var(nc, 'disk_u', r['disk_u'], ('frame', 'walker'),
             'walker_disk_u', 'Walker u coordinate in the Poincare disk')
        _var(nc, 'disk_v', r['disk_v'], ('frame', 'walker'),
             'walker_disk_v', 'Walker v coordinate in the Poincare disk')
        _var(nc, 'dist_from_start', r['dist_from_start'], ('frame', 'walker'),
             'fisher_rao_distance', 'Fisher-Rao distance of each walker from its origin')
        _var(nc, 'entropy_series', r['entropy_series'], ('frame',),
             'ensemble_entropy', 'Differential entropy estimate of the walker ensemble')
        _var(nc, 'ref_geodesic_disk', r['ref_geodesic_disk'], ('ref_sample', 'plane'),
             'reference_geodesic', 'Vertical reference geodesic from the start, disk coords')

    # ------------------------------------------------------------------- CSV
    @staticmethod
    def save_csv(filename, result, metadata, output_dir="outputs"):
        """Write the full metric table for one case to CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename

        row = {'scenario': metadata.get('scenario_name', 'unknown'),
               'case_type': metadata.get('case_type', 'unknown'),
               'kind': result['kind']}
        for k, v in result['metrics'].items():
            if not isinstance(v, (tuple, list, np.ndarray)):
                row[k] = v
        pd.DataFrame([row]).to_csv(filepath, index=False)
        return str(filepath)

    @staticmethod
    def _comparison_summary(result, metadata):
        """Curated cross-case comparison row (heterogeneous cases made comparable)."""
        kind = result['kind']
        me = result['metrics']
        table = {
            'tessellation': ('modular_relations_residual', 'n_tiles'),
            'sphere': ('geodesic_length_residual', 'entropy_max'),
            'dual_weave': ('pythagorean_residual', 'kl_asymmetry_max'),
            'diffusion': ('martingale_deviation', 'escape_rate'),
        }
        res_name, con_name = table[kind]
        return {
            'scenario': metadata.get('scenario_name', 'unknown'),
            'case_type': metadata.get('case_type', 'unknown'),
            'kind': kind,
            'curvature': me.get('curvature', float('nan')),
            'correctness_metric': res_name,
            'correctness_value': me.get(res_name, float('nan')),
            'content_metric': con_name,
            'content_value': me.get(con_name, float('nan')),
        }

    @staticmethod
    def append_comparison_csv(filename, result, metadata, output_dir="outputs"):
        """Append a curated comparison row for multi-case analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename

        df_new = pd.DataFrame([DataHandler._comparison_summary(result, metadata)])
        if filepath.exists():
            df = pd.concat([pd.read_csv(filepath), df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(filepath, index=False)
        return str(filepath)
