"""Simple timer for performance tracking."""

import time
from contextlib import contextmanager
from typing import Dict


class Timer:
    """Performance timer with section tracking."""

    def __init__(self):
        self.times: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.time()

    def stop(self, name: str) -> float:
        """Stop timing a section."""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0.0

    @contextmanager
    def time_section(self, name: str):
        """Context manager for timing a code block."""
        self.start(name)
        yield
        self.stop(name)

    def get_times(self) -> Dict[str, float]:
        """Get all recorded times."""
        return self.times.copy()

    def reset(self):
        """Reset all timers."""
        self.times = {}
        self.start_times = {}

    def total(self) -> float:
        """Get total elapsed time across all sections."""
        return sum(self.times.values())

    def summary(self) -> str:
        """Get formatted timing summary."""
        lines = ["Timing Summary:"]
        for name, elapsed in sorted(self.times.items()):
            lines.append(f"  {name}: {elapsed:.3f}s")
        lines.append(f"  TOTAL: {self.total():.3f}s")
        return "\n".join(lines)
