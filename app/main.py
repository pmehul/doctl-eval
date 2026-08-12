"""The web API.

The app *is* the test. The numbers in the UI come out of the same code you can run
yourself, not a spreadsheet pasted into a template.

One run at a time, held by a lock. Two at once would share the same concurrency
limit and both would report latency neither of them actually saw.

On security, and this is a deployment choice rather than something I forgot: these
endpoints have no login. That's fine for a single-user tool on localhost or behind
an authenticating proxy. It is not fine on a public address, because POST /api/run
spends real money using the key in the container's environment. Exposed publicly it
would need a login in front of it and a spend cap per caller.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import catalog, store
from .config import settings
from .corpus import LABELS, load_corpus
from .prompting import PROMPT_VERSION, SYSTEM_PROMPT, build_messages, few_shot_messages
from .runner import RunProgress, execute_run

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="doctl issue-classification eval harness", version="1.0.0")

_run_lock = asyncio.Lock()
_current: RunProgress | None = None
_last_result: dict[str, Any] | None = None


@app.get("/api/health")
async def health() -> dict[str, Any]:
    problems = settings.validate()
    corpus_ok, corpus_err = True, None
    try:
        load_corpus()
    except SystemExit as exc:
        corpus_ok, corpus_err = False, str(exc)
    return {
        "ok": not problems and corpus_ok,
        "provider": settings.provider,
        "simulated": settings.is_mock,
        "config_problems": problems,
        "corpus_ok": corpus_ok,
        "corpus_error": corpus_err,
        "prompt_version": PROMPT_VERSION,
        "defaults": {
            "concurrency": settings.concurrency,
            "scored_split": settings.scored_split,
            "temperature": settings.temperature,
            "request_timeout_s": settings.request_timeout_s,
            "max_retries": settings.max_retries,
        },
    }


@app.get("/api/catalog")
async def get_catalog() -> dict[str, Any]:
    return catalog.catalog_payload()


@app.get("/api/corpus")
async def get_corpus() -> dict[str, Any]:
    corpus = load_corpus()
    return {
        "repo": corpus.repo,
        "corpus_hash": corpus.corpus_hash,
        "frozen_at": corpus.frozen_at,
        "labels": list(LABELS),
        "n_issues": len(corpus.issues),
        "n_workload": len(corpus.workload),
        "n_scored_test": len(corpus.scored("test")),
        "n_scored_dev": len(corpus.scored("dev")),
        "n_unscored": len(corpus.unscored()),
        "gold": corpus.gold_stats,
    }


@app.get("/api/prompt")
async def get_prompt() -> dict[str, Any]:
    """Expose the exact prompt. The prompt is the experiment's main confound, so
    it should be readable without cloning the repo."""
    corpus = load_corpus()
    sample = next((i for i in corpus.workload if i.gold_split == "test"), corpus.workload[0])
    return {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "few_shot": [dict(m) for m in few_shot_messages()],
        "example_rendered_request": build_messages(sample),
        "example_issue_number": sample.number,
    }


@app.get("/api/runs")
async def get_runs() -> dict[str, Any]:
    return {"runs": store.list_runs(settings.runs_dir)}


@app.get("/api/runs/{filename}")
async def get_run(filename: str) -> Any:
    try:
        return store.load_run(settings.runs_dir, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.get("/api/progress")
async def get_progress() -> dict[str, Any]:
    if _current is None:
        return {"state": "idle"}
    return _current.payload()


@app.get("/api/result")
async def get_result() -> Any:
    if _last_result is None:
        raise HTTPException(status_code=404, detail="no run has completed in this process yet")
    return _last_result


@app.post("/api/run")
async def post_run(payload: dict[str, Any] = Body(default={})) -> Any:
    global _current, _last_result

    if _run_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="a run is already in progress; poll /api/progress",
        )

    problems = settings.validate()
    if problems:
        raise HTTPException(status_code=400, detail={"config_problems": problems})

    model_a = payload.get("model_a") or catalog.DEFAULT_MODEL_A
    model_b = payload.get("model_b") or catalog.DEFAULT_MODEL_B
    for model_id in (model_a, model_b):
        if model_id not in catalog.BY_ID:
            raise HTTPException(status_code=400, detail=f"unknown model {model_id!r}")

    try:
        concurrency = int(payload.get("concurrency") or settings.concurrency)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="concurrency must be an integer") from None
    if not 1 <= concurrency <= 256:
        raise HTTPException(status_code=400, detail="concurrency must be between 1 and 256")

    scored_split = (payload.get("scored_split") or settings.scored_split).lower()
    if scored_split not in {"test", "dev", "all"}:
        raise HTTPException(status_code=400, detail="scored_split must be test|dev|all")

    corpus = load_corpus()
    total_calls = len(corpus.workload) * 2

    async with _run_lock:
        _current = RunProgress(run_id=uuid.uuid4().hex[:12], total=total_calls)
        try:
            result = await execute_run(
                settings=settings,
                model_a=model_a,
                model_b=model_b,
                concurrency=concurrency,
                scored_split=scored_split,
                progress=_current,
            )
        except Exception as exc:  # surfaced to the UI rather than swallowed
            _current.state = "failed"
            _current.message = f"{type(exc).__name__}: {exc}"
            raise HTTPException(status_code=500, detail=_current.message) from exc

        _current.state = "done"
        _last_result = result
        if settings.persist_runs:
            path = store.save_run(settings.runs_dir, result)
            result["persisted_to"] = path.name

    return result


# --- static UI ------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> Any:
    idx = STATIC_DIR / "index.html"
    if not idx.is_file():
        return JSONResponse({"detail": "UI not built"}, status_code=500)
    return FileResponse(str(idx))
