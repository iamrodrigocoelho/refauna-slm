"""Amostragem de RAM: processo do app (+ filhos), processos do Ollama e sistema (pico em GB)."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil

GB = 1024**3


def _rss(procs) -> int:
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


UNITS = {"B": 1 / GB, "KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1.0}


def footprint_gb(pids: list[int]) -> float:
    """Memória física (phys_footprint) via /usr/bin/footprint do macOS: inclui GPU/Metal, que o RSS não vê."""
    if not pids:
        return 0.0
    try:
        out = subprocess.run(["footprint", *map(str, pids)], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    return sum(float(v) * UNITS[u] for v, u in re.findall(r"^\S.*\]: .*Footprint: ([\d.]+) (B|KB|MB|GB)", out, re.M))


def ollama_procs() -> list[psutil.Process]:
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            cmd = " ".join(p.info["cmdline"] or []).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "ollama" in name or "ollama" in cmd.split(" ")[0]:
            out.append(p)
    return out


@dataclass
class RamPeak:
    app_gb: float = 0.0
    ollama_gb: float = 0.0
    total_gb: float = 0.0  # app + Ollama no mesmo instante (RSS)
    app_footprint_gb: float = 0.0
    ollama_footprint_gb: float = 0.0
    footprint_total_gb: float = 0.0  # app + Ollama no mesmo instante (phys_footprint)
    system_used_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_start_gb: float = 0.0
    samples: int = 0
    timeline: list[tuple[float, float, float]] = field(default_factory=list)  # (t, app, ollama) em footprint


class RamSampler(threading.Thread):
    """Mede em segundo plano. `pid` = processo do app (inclui filhos)."""

    def __init__(self, pid: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.peak = RamPeak(swap_start_gb=psutil.swap_memory().used / GB)
        self._halt = threading.Event()
        self._t0 = time.time()

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(self.interval)

    def sample(self) -> None:
        try:
            app_procs = [self.proc, *self.proc.children(recursive=True)]
        except psutil.NoSuchProcess:
            app_procs = []
        oll_procs = ollama_procs()
        app, oll = _rss(app_procs) / GB, _rss(oll_procs) / GB
        app_fp, oll_fp = footprint_gb([p.pid for p in app_procs]), footprint_gb([p.pid for p in oll_procs])
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        p = self.peak
        p.app_gb, p.ollama_gb = max(p.app_gb, app), max(p.ollama_gb, oll)
        p.total_gb = max(p.total_gb, app + oll)
        p.app_footprint_gb = max(p.app_footprint_gb, app_fp)
        p.ollama_footprint_gb = max(p.ollama_footprint_gb, oll_fp)
        p.footprint_total_gb = max(p.footprint_total_gb, app_fp + oll_fp)
        p.system_used_gb = max(p.system_used_gb, (vm.total - vm.available) / GB)
        p.swap_used_gb = max(p.swap_used_gb, sw.used / GB)
        p.samples += 1
        p.timeline.append((round(time.time() - self._t0, 1), round(app_fp, 2), round(oll_fp, 2)))

    def stop(self) -> RamPeak:
        self._halt.set()
        self.join(timeout=5)
        return self.peak
