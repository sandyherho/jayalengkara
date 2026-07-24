"""Simulation logger for Fisher-Rao information-geometry runs."""

import logging
from pathlib import Path
from typing import Dict, Any


class SimulationLogger:
    """Logger for Fisher-Rao case runs."""

    def __init__(self, scenario_name: str, log_dir: str = "logs",
                 verbose: bool = True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"

        self.logger = self._setup_logger()
        self.warnings = []
        self.errors = []

    def _setup_logger(self):
        logger = logging.getLogger(f"fr_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)
        self.warnings.append(msg)
        if self.verbose:
            print(f"  WARNING: {msg}")

    def error(self, msg: str):
        self.logger.error(msg)
        self.errors.append(msg)
        if self.verbose:
            print(f"  ERROR: {msg}")

    def log_parameters(self, params: Dict[str, Any]):
        self.info("=" * 60)
        self.info(f"PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        self.info("=" * 60)

    def log_timing(self, timing: Dict[str, float]):
        self.info("=" * 60)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 60)
        for key, value in sorted(timing.items()):
            self.info(f"  {key}: {value:.3f} s")
        self.info("=" * 60)

    def log_results(self, result: Dict[str, Any]):
        self.info("=" * 60)
        self.info("RUN RESULTS")
        self.info("=" * 60)
        self.info(f"  case: {result.get('case')}")
        self.info(f"  kind: {result.get('kind')}")
        self.info("  Diagnostic metrics:")
        for k, v in result['metrics'].items():
            self.info(f"    {k}: {v}")
        self.info("=" * 60)

    def finalize(self):
        self.info("=" * 60)
        self.info("RUN SUMMARY")
        self.info("=" * 60)
        if self.errors:
            self.info(f"  ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"    {i}. {err}")
        else:
            self.info("  ERRORS: None")
        if self.warnings:
            self.info(f"  WARNINGS: {len(self.warnings)}")
        else:
            self.info("  WARNINGS: None")
        self.info(f"  Log file: {self.log_file}")
        self.info("=" * 60)
        self.info(f"Run completed: {self.scenario_name}")
        self.info("=" * 60)
