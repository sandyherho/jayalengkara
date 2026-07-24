#!/usr/bin/env python
"""
Command-line interface for jayalengkara.

Four default cases realize distinct Fisher-Rao geometries for comparative study.
Each run writes a NetCDF archive, a CSV metric table, a PNG diagnostic panel, and
an animated GIF.
"""

import argparse
import sys
from pathlib import Path

from .core.models import FisherRaoModel
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer

__version__ = "0.0.1"

CASE_MAP = {
    'case1': 'case1_gaussian_hyperbolic',
    'case2': 'case2_categorical_sphere',
    'case3': 'case3_dual_weave',
    'case4': 'case4_hyperbolic_diffusion',
}


def print_header():
    print("\n" + "=" * 70)
    print(" " * 12 + "jayalengkara: Fisher-Rao Information Geometry")
    print(" " * 20 + "as a Computational Medium")
    print(" " * 28 + "Version " + __version__)
    print("=" * 70)
    print("\n  Exact geodesics, curvatures, and divergences")
    print("  Numba JIT acceleration + parallel processing")
    print("\n  License: MIT")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    clean = scenario_name.lower().replace(' - ', '_').replace('-', '_').replace(' ', '_')
    while '__' in clean:
        clean = clean.replace('__', '_')
    if clean.startswith('case_'):
        parts = clean.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            clean = f"case{parts[1]}_" + '_'.join(parts[2:])
    return clean.strip('_')


def run_scenario(config: dict, output_dir: str = "outputs",
                 verbose: bool = True, n_cores=None):
    """Run a complete Fisher-Rao case: compute, archive, and visualize."""
    config = ConfigManager.validate_config(config)
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name}  [{config.get('case_type')}]")
        print(f"{'=' * 60}")

    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")

    try:
        logger.log_parameters(config)

        with timer.time_section("model_init"):
            if verbose:
                print("\n[1/4] Initializing model...")
            cores = n_cores if n_cores is not None else config.get('n_cores', 0)
            model = FisherRaoModel(n_cores=cores, verbose=verbose, logger=logger)

        with timer.time_section("compute"):
            if verbose:
                print("\n[2/4] Computing geometry...")
            result = model.run(config)
            logger.log_results(result)

        if config.get('save_netcdf', True) or config.get('save_csv', True):
            with timer.time_section("save_data"):
                if verbose:
                    print("\n[3/4] Saving data...")
                if config.get('save_netcdf', True):
                    nc_file = f"{clean_name}.nc"
                    DataHandler.save_netcdf(nc_file, result, config, output_dir)
                    if verbose:
                        print(f"      Saved: {output_dir}/{nc_file}")
                if config.get('save_csv', True):
                    csv_file = f"{clean_name}_metrics.csv"
                    DataHandler.save_csv(csv_file, result, config, output_dir)
                    DataHandler.append_comparison_csv(
                        "comparison_metrics.csv", result, config, output_dir)
                    if verbose:
                        print(f"      Saved: {output_dir}/{csv_file}")

        if config.get('save_animation', True) or config.get('save_diagnostics', True):
            with timer.time_section("visualization"):
                if verbose:
                    print("\n[4/4] Creating visualizations...")
                if config.get('save_diagnostics', True):
                    diag_file = f"{clean_name}_diagnostics.png"
                    Animator.create_diagnostics(
                        result, diag_file, output_dir, dpi=config.get('dpi', 150))
                if config.get('save_animation', True):
                    gif_file = f"{clean_name}.gif"
                    Animator.create_gif(
                        result, gif_file, output_dir,
                        fps=config.get('fps', 30), dpi=config.get('dpi', 150),
                        colormap=config.get('colormap', 'twilight'),
                        marker_size=config.get('marker_size', 8),
                        alpha=config.get('alpha', 0.85))

        timer.stop("total")
        logger.log_timing(timer.get_times())

        if verbose:
            print(f"\n{'=' * 60}")
            print("RUN COMPLETED SUCCESSFULLY")
            print(f"  Compute time: {timer.times.get('compute', 0):.2f} s")
            print(f"  Visualization time: {timer.times.get('visualization', 0):.2f} s")
            print(f"  Total time: {timer.times.get('total', 0):.2f} s")
            print(f"{'=' * 60}\n")
        return result

    except Exception as e:
        logger.error(f"Run failed: {str(e)}")
        if verbose:
            print(f"\n{'=' * 60}\nRUN FAILED\n  Error: {str(e)}\n{'=' * 60}\n")
        raise
    finally:
        logger.finalize()


def main():
    parser = argparse.ArgumentParser(
        description='jayalengkara: Fisher-Rao information geometry as a medium',
        epilog='Example: jayalengkara case3 --cores 8')
    parser.add_argument('case', nargs='?',
                        choices=['case1', 'case2', 'case3', 'case4'],
                        help='Default case to run (case1-4)')
    parser.add_argument('--config', '-c', type=str,
                        help='Path to a custom configuration file')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run all four default cases sequentially')
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                        help='Output directory (default: outputs)')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of CPU cores (default: all available)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (minimal output)')
    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print_header()

    configs_dir = Path(__file__).parent.parent.parent / 'configs'

    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)

    elif args.all:
        config_files = sorted(configs_dir.glob('case*.txt'))
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            sys.exit(1)
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] {cfg_file.stem}...")
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        if verbose:
            print(f"\nComparison CSV: {args.output_dir}/comparison_metrics.csv")

    elif args.case:
        cfg_file = configs_dir / f"{CASE_MAP[args.case]}.txt"
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"ERROR: Configuration file not found: {cfg_file}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
