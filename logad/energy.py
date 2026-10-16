"""Cost measurement: CPU time (measured), energy (measured or estimated).

Energy is read from Intel RAPL (/sys/class/powercap) when the current user is
allowed to; since 2020 most Linux systems restrict it to root. Otherwise it is
estimated as CPU-seconds x average power per busy core, which is the approach
tools such as CodeCarbon fall back to. Every result records which one was used.

Runs are single-threaded (set in run.py), so CPU time is comparable across
detectors and the estimate is proportional to it.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
# Lab laptop: Intel i5-8350U, 15 W TDP, 4 cores -> 3.75 W per busy core. Override with
# LOGAD_WATTS_PER_CORE for another machine.
WATTS_PER_CORE = float(os.environ.get("LOGAD_WATTS_PER_CORE", 15.0 / 4))


def rapl_readable():
    try:
        int(RAPL.read_text())
        return True
    except (OSError, ValueError):
        return False


class Meter:
    def __init__(self):
        self.cpu_s = 0.0
        self.wall_s = 0.0
        self.rapl_j = None

    @property
    def energy_j(self):
        return self.rapl_j if self.rapl_j is not None else self.cpu_s * WATTS_PER_CORE

    @property
    def energy_source(self):
        return "measured (RAPL package)" if self.rapl_j is not None else f"estimated ({WATTS_PER_CORE:.2f} W x CPU-s)"


@contextmanager
def measure():
    m = Meter()
    use_rapl = rapl_readable()
    e0 = int(RAPL.read_text()) if use_rapl else None
    c0, w0 = time.process_time(), time.perf_counter()
    try:
        yield m
    finally:
        m.cpu_s = time.process_time() - c0
        m.wall_s = time.perf_counter() - w0
        if use_rapl:
            e1 = int(RAPL.read_text())
            wrap_file = RAPL.with_name("max_energy_range_uj")
            wrap = int(wrap_file.read_text()) if wrap_file.exists() else 1 << 32
            m.rapl_j = ((e1 - e0) % wrap) / 1e6   # package-wide: includes other processes
