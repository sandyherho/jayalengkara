"""
jayalengkara: Fisher-Rao Information Geometry as a Computational Medium

A high-performance Python library that renders the Fisher-Rao geometry of
elementary statistical families as archivable, animated geometry. Four default
cases realize the Gaussian manifold as the hyperbolic plane, the categorical
manifold as the sphere, the dually flat weave of the Gaussian exponential
family, and Brownian motion under the Fisher-Rao metric. Numerical kernels are
accelerated and parallelized with Numba.
"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho"
__license__ = "MIT"

from .core.models import FisherRaoModel, CASE_KINDS
from .core import geometry, cases, diagnostics
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator

__all__ = [
    "FisherRaoModel",
    "CASE_KINDS",
    "geometry",
    "cases",
    "diagnostics",
    "ConfigManager",
    "DataHandler",
    "Animator",
]
