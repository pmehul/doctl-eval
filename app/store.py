"""Saving runs to disk.

Every finished run goes to one JSON file with the detail for every issue, not just
the totals. That costs a few megabytes a run and buys the thing that matters in a
review: any number on screen can be traced back to the calls behind it, long after
the process has exited.

The filename carries the run id, both model names and the dataset fingerprint, so
listing the directory tells you what was compared against which data without
opening anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SAFE.sub("-", text)[:48]


def save_run(runs_dir: Path, payload: dict[str, Any]) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    a = _slug(payload["models"]["a"]["id"])
    b = _slug(payload["models"]["b"]["id"])
    corpus_hash = payload["corpus"]["corpus_hash"]
    tag = "sim" if payload.get("simulated") else "live"
    path = runs_dir / f"run-{payload['run_id']}-{tag}-{a}__vs__{b}-{corpus_hash}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Index of persisted runs. Reads each file's summary, skipping item detail."""
    if not runs_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("run-*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "run_id": payload.get("run_id"),
                "file": path.name,
                "finished_at": payload.get("finished_at"),
                "simulated": payload.get("simulated", False),
                "model_a": payload.get("models", {}).get("a", {}).get("id"),
                "model_b": payload.get("models", {}).get("b", {}).get("id"),
                "corpus_hash": payload.get("corpus", {}).get("corpus_hash"),
                "scored_split": payload.get("corpus", {}).get("scored_split"),
                "concurrency": payload.get("config", {}).get("concurrency"),
                "a_macro_f1": payload.get("scored", {}).get("a", {}).get("macro_f1"),
                "b_macro_f1": payload.get("scored", {}).get("b", {}).get("macro_f1"),
                "a_accuracy": payload.get("scored", {}).get("a", {}).get("accuracy"),
                "b_accuracy": payload.get("scored", {}).get("b", {}).get("accuracy"),
                "wall_clock_s": payload.get("operational", {}).get("wall_clock_s"),
            }
        )
    return out


def load_run(runs_dir: Path, filename: str) -> dict[str, Any]:
    # Guard against path traversal: only plain filenames inside runs_dir.
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError("invalid run filename")
    path = runs_dir / filename
    if not path.is_file():
        raise FileNotFoundError(filename)
    return json.loads(path.read_text(encoding="utf-8"))
