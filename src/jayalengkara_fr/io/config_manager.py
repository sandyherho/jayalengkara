"""Configuration parser for Fisher-Rao information-geometry cases."""

from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Parse plain-text configuration files and supply embedded defaults."""

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a plain-text file.

        File format:
            # Comments
            key = value

        Supported value types: bool, int, float, str.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if '#' in value:
                    value = value.split('#')[0].strip()
                config[key] = ConfigManager._parse_value(value)
        return config

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse a string into bool, int, float, or str."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill embedded defaults for any keys the file omits.

        The defaults span every case; a given case reads only the keys it needs,
        so a single embedded default table serves all four scenarios.
        """
        defaults = {
            # Scenario selection
            'scenario_name': 'Fisher-Rao Case',
            'case_type': 'gaussian_hyperbolic',

            # Parallelism and output
            'n_cores': 0,
            'output_dir': 'outputs',
            'save_netcdf': True,
            'save_csv': True,
            'save_animation': True,
            'save_diagnostics': True,

            # Animation and rendering
            'n_frames': 120,
            'fps': 30,
            'dpi': 150,
            'colormap': 'jl_aurora',
            'marker_size': 8,
            'alpha': 0.85,
            'seed': 42,

            # Case 1: gaussian_hyperbolic
            'max_tiles': 160,
            'edge_samples': 48,
            'field_res': 220,

            # Case 2: categorical_sphere
            'subdivision_depth': 3,
            'n_sample_geodesics': 4,
            'n_orbit_points': 28,

            # Case 3: dual_weave
            'n_lines': 9,
            'line_samples': 60,
            'P_mu': -1.0, 'P_sigma': 0.7,
            'Q_mu': 0.6, 'Q_sigma': 1.1,
            'loop_mu_amp': 0.9, 'loop_sigma_amp': 0.35,

            # Case 4: hyperbolic_diffusion
            'mu0': 0.0, 'sigma0': 1.0,
            'n_walkers': 400, 'substeps': 6, 'dt': 0.01,
        }
        for key, default in defaults.items():
            if key not in config:
                config[key] = default
        return config
