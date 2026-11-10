#!/usr/bin/env python3
"""
Jetson Nano rover startup script.

Starts all rover-side processes in the correct order:
  1. motor-controller gRPC service (port 50052)
  2. stereo-perception pipeline (main perception + control + Kafka publisher)

Usage:
  python3 scripts/run_rover.py [--config config/rover.yaml] [--dry-run]

Environment variables are loaded from .env.rover if present.
All processes are supervised: if one crashes it is restarted.

For production deployment, use the systemd units in deployment/jetson/.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_rover")

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.join(_HERE, "..")
_VENV_PY  = os.path.join(_ROOT, "venv", "bin", "python3")
_PYTHON   = _VENV_PY if os.path.exists(_VENV_PY) else sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default=os.path.join(_ROOT, "config", "rover.yaml"))
    p.add_argument("--env-file", default=os.path.join(_ROOT, ".env.rover"))
    p.add_argument("--dry-run",  action="store_true", help="Print commands without running")
    return p.parse_args()


def load_env_file(path: str) -> dict:
    env = {}
    if not os.path.exists(path):
        return env
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


class ProcessManager:
    """Simple supervisor: start processes, watch them, restart on crash."""

    def __init__(self, env: dict, dry_run: bool = False) -> None:
        self._processes: list[dict] = []
        self._env     = {**os.environ, **env}
        self._dry_run = dry_run
        self._stop    = False

    def add(self, name: str, cmd: list[str], cwd: str = _ROOT,
            restart_delay: float = 2.0) -> None:
        self._processes.append({
            "name":          name,
            "cmd":           cmd,
            "cwd":           cwd,
            "restart_delay": restart_delay,
            "proc":          None,
        })

    def run(self) -> None:
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Start all processes
        for p in self._processes:
            self._start(p)
            time.sleep(1.0)  # stagger startups

        log.info("All rover processes started. Ctrl+C to stop.")

        try:
            while not self._stop:
                for p in self._processes:
                    if p["proc"] and p["proc"].poll() is not None:
                        rc = p["proc"].returncode
                        log.warning("Process %r exited (rc=%d); restarting in %.1fs ...",
                                    p["name"], rc, p["restart_delay"])
                        time.sleep(p["restart_delay"])
                        if not self._stop:
                            self._start(p)
                time.sleep(1.0)
        finally:
            self._stop_all()

    def _start(self, p: dict) -> None:
        log.info("Starting: %s  %s", p["name"], " ".join(p["cmd"]))
        if self._dry_run:
            p["proc"] = None
            return
        p["proc"] = subprocess.Popen(
            p["cmd"], cwd=p["cwd"], env=self._env,
            stdout=None, stderr=None,  # inherit parent stdout/stderr
        )

    def _stop_all(self) -> None:
        for p in reversed(self._processes):
            if p["proc"] and p["proc"].poll() is None:
                log.info("Stopping %s ...", p["name"])
                p["proc"].terminate()
        time.sleep(2.0)
        for p in self._processes:
            if p["proc"] and p["proc"].poll() is None:
                p["proc"].kill()

    def _handle_signal(self, sig, _) -> None:
        log.info("Signal %s: shutting down rover", sig)
        self._stop = True


def main() -> None:
    args = parse_args()
    env  = load_env_file(args.env_file)

    if env:
        log.info("Loaded env from %s (%d variables)", args.env_file, len(env))

    manager = ProcessManager(env, dry_run=args.dry_run)

    # 1. Motor controller gRPC service
    manager.add(
        name = "motor-controller",
        cmd  = [_PYTHON, "-m", "services.motor-controller.main"],
        # Alternative: run directly
        # cmd = [_PYTHON, os.path.join(_ROOT, "services/motor-controller/main.py")],
    )

    # 2. Stereo perception pipeline (includes control loop + Kafka publisher)
    manager.add(
        name = "stereo-perception",
        cmd  = [_PYTHON, os.path.join(_ROOT, "services/stereo-perception/main.py")],
    )

    if args.dry_run:
        log.info("Dry run — commands that would run:")
        for p in manager._processes:
            log.info("  [%s]  %s", p["name"], " ".join(p["cmd"]))
        return

    manager.run()


if __name__ == "__main__":
    main()
