"""Core Fisher-Rao geometry, case generators, model engine, and diagnostics."""

from . import geometry
from . import cases
from . import diagnostics
from .models import FisherRaoModel, CASE_KINDS

__all__ = ["geometry", "cases", "diagnostics", "FisherRaoModel", "CASE_KINDS"]
