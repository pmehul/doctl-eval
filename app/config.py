"""Settings, all read from the environment.

Anything you'd want to change between runs is an environment variable, so the same
image runs at concurrency 4 on a laptop and 64 against real capacity with no
rebuild. The exercise asks for that, and it's how it should work anyway. Having to
rebuild a container to change a worker count is a smell.

CONCURRENCY can also be set per run from the UI, because trying a few values to
find where throughput stops improving is the whole reason p95 is reported next to
the concurrency it was measured at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- provider ---------------------------------------------------------
    # "digitalocean" makes real calls. "mock" runs the whole thing offline against
    # a simulator, so the UI, the scoring, the concurrency limit and the cost maths
    # can all be tried out without spending credits. Which mode is on comes back in
    # every API response and gets a banner in the UI, because a fake number that
    # reads like a real one is worse than no number.
    provider: str = field(default_factory=lambda: os.environ.get("PROVIDER", "digitalocean").lower())
    api_key: str = field(default_factory=lambda: os.environ.get("DO_INFERENCE_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "DO_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1"
        ).rstrip("/")
    )

    # --- concurrency and timeouts ----------------------------------------
    # 8 by default. Enough to keep things busy across 536 issues without leaning on
    # a shared tier. Going higher buys wall-clock time and pays for it in p95, since
    # queue time shows up as latency, and in rate limits. The UI lets you find where
    # that stops being worth it instead of guessing.
    concurrency: int = field(default_factory=lambda: _int("CONCURRENCY", 8))
    request_timeout_s: float = field(default_factory=lambda: _float("REQUEST_TIMEOUT_S", 60.0))
    max_retries: int = field(default_factory=lambda: _int("MAX_RETRIES", 3))
    retry_base_delay_s: float = field(default_factory=lambda: _float("RETRY_BASE_DELAY_S", 1.0))

    # --- generation -------------------------------------------------------
    # Temperature 0. This is sorting, not writing. Any randomness would turn up as
    # accuracy wobbling between runs, and then nothing reproduces.
    temperature: float = field(default_factory=lambda: _float("TEMPERATURE", 0.0))
    # Enough for a short JSON object. Reasoning models need far more room, because
    # they write out their thinking first. See catalog.reasoning.
    max_tokens: int = field(default_factory=lambda: _int("MAX_TOKENS", 96))
    reasoning_max_tokens: int = field(default_factory=lambda: _int("REASONING_MAX_TOKENS", 1400))

    # --- corpus -----------------------------------------------------------
    corpus_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("CORPUS_PATH", str(ROOT / "data" / "corpus" / "doctl-issues-snapshot.json"))
        )
    )
    gold_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("GOLD_PATH", str(ROOT / "data" / "ground_truth" / "gold.json"))
        )
    )
    runs_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("RUNS_DIR", str(ROOT / "data" / "runs")))
    )

    # Which half of the answer key the scored view uses. "test" by default. "dev" is
    # where I tuned the prompt, so its scores are flattering and shouldn't be quoted.
    scored_split: str = field(default_factory=lambda: os.environ.get("SCORED_SPLIT", "test").lower())

    # Classify fewer issues for a quick test, without touching the saved dataset.
    max_issues: int = field(default_factory=lambda: _int("MAX_ISSUES", 0))

    persist_runs: bool = field(default_factory=lambda: _bool("PERSIST_RUNS", True))

    # Simulator only. It waits its fake latency divided by this, so the concurrency
    # limit still has to queue work and throughput and wall clock come out the right
    # shape, without a demo taking two minutes. Set it to 1 for an unshrunk clock.
    mock_time_scale: float = field(default_factory=lambda: _float("MOCK_TIME_SCALE", 10.0))

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.provider not in {"digitalocean", "mock"}:
            problems.append(f"PROVIDER must be 'digitalocean' or 'mock', got {self.provider!r}")
        if self.provider == "digitalocean" and not self.api_key:
            problems.append(
                "DO_INFERENCE_API_KEY is not set. Set it, or run with PROVIDER=mock "
                "to exercise the harness offline."
            )
        if self.concurrency < 1:
            problems.append("CONCURRENCY must be >= 1")
        if self.scored_split not in {"test", "dev", "all"}:
            problems.append("SCORED_SPLIT must be one of test|dev|all")
        if self.mock_time_scale <= 0:
            problems.append("MOCK_TIME_SCALE must be > 0")
        return problems


settings = Settings()
