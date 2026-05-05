"""Per-run metadata snapshots and per-patient failure capture.

Pipeline stage: cross-cutting; called from every script's ``main()``.

Inputs
------
* ``phase`` -- short name like ``"preprocessing"`` or ``"tcav"``.
* ``cfg``   -- the full YAML config dict.
* ``args``  -- parsed CLI arguments (already converted via ``vars(args)``).

Outputs
-------
* ``logs/run_<phase>_<timestamp>.json`` -- one snapshot per script run.
* ``logs/<phase>_failures.jsonl`` -- one JSON event per failed patient.

Side effects
------------
* Creates the ``logs/`` directory and parent paths on demand.
* Best-effort ``git rev-parse HEAD`` capture (no failure propagated).

Failure modes
-------------
* Missing ``logs_dir`` in the config -> falls back to ``./logs``.
* ``git`` not on PATH -> ``git_sha`` recorded as ``"unknown"``.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe(callable_, default):
    try:
        return callable_()
    except Exception:  # noqa: BLE001
        return default


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode("utf-8").strip() or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _torch_versions() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = (
            torch.version.cuda if torch.cuda.is_available() else None
        )
        info["device_count"] = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    return info


_ALLOWED_ENV_PREFIXES: tuple[str, ...] = ("HCC_", "CUDA_", "PYTHON")


def _get_gpu_memory_gb() -> float:
    """Return GPU memory currently reserved (GB), or 0.0 when unavailable."""
    try:
        import torch

        return torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _resolve_logs_dir(cfg: dict | None) -> Path:
    if cfg is None:
        return Path("logs")
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    return Path(paths.get("logs_dir", "logs"))


def record_run(
    phase: str,
    cfg: dict | None,
    args: dict[str, Any] | None = None,
    ) -> Path:
    """
    Snapshot environment + config at the start of a script run.

    Returns
    -------
    Path
        Location of the JSON snapshot.
    """
    logs_dir = _resolve_logs_dir(cfg)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = logs_dir / f"run_{phase}_{ts}.json"
    safe_env = {
        k: v for k, v in os.environ.items()
        if k.startswith(_ALLOWED_ENV_PREFIXES)
    }
    safe_cfg = dict(cfg) if cfg else {}
    if "credentials" in safe_cfg:
        safe_cfg["credentials"] = "REDACTED"

    payload = {
        "phase": phase,
        "timestamp": ts,
        "git_sha": _git_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": _safe(socket.gethostname, "unknown"),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "cli_args": args or {},
        "cwd": str(Path.cwd()),
        "env": safe_env,
        "config": safe_cfg,
    }
    payload.update(_torch_versions())
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Saved run metadata -> %s", out)
    except OSError as exc:
        logger.warning("Could not write run metadata to %s: %s", out, exc)
    return out


def record_failure(
    phase: str,
    patient_id: str,
    exc: BaseException,
    extra: dict[str, Any] | None = None,
    logs_dir: Path | str | None = None,
) -> Path:
    """Append a per-patient failure record to ``logs/<phase>_failures.jsonl``."""
    base = Path(logs_dir or "logs")
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{phase}_failures.jsonl"
    event = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase,
        "patient_id": patient_id,
        "error_type": type(exc).__name__,
        "error_msg": str(exc),
        "gpu_memory_gb": _get_gpu_memory_gb(),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    if extra:
        event["extra"] = extra
    try:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        logger.warning("Recorded failure for %s -> %s", patient_id, out)
    except OSError as exc2:
        logger.error("Could not write failure record: %s", exc2)
    return out
