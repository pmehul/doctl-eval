"""The web API.

The app *is* the test. The numbers in the UI come out of the same code you can run
yourself, not a spreadsheet pasted into a template.

One run at a time, held by a lock. Two at once would share the same concurrency
limit and both would report latency neither of them actually saw.

Everything is behind HTTP Basic auth when BASIC_AUTH_PASSWORD is set. The reason is
money rather than privacy: POST /api/run spends real credits using the key in the
container's environment, and a full run costs roughly $0.27, or about $1.93 if the
caller picks a reasoning model for both slots. With no login and no rate limit, a
public URL could drain the whole credit balance in an afternoon. So the password is
required whenever the app is reachable from outside localhost.

It is deliberately Basic auth and nothing cleverer. There is one user, the browser
handles the prompt natively, and it costs no extra dependency. It is not a real
authorisation system: there are no accounts, no per-caller spend caps, and no audit
of who ran what. For a single-user evaluation tool behind TLS that is the right
amount of security. For anything multi-tenant it would not be.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from . import catalog, store
from .config import settings
from .corpus import LABELS, load_corpus
from .prompting import PROMPT_VERSION, SYSTEM_PROMPT, build_messages, few_shot_messages
from .runner import RunProgress, execute_run

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="doctl issue-classification eval harness", version="1.0.0")

# --- authentication -------------------------------------------------------

BASIC_AUTH_USERNAME = os.environ.get("BASIC_AUTH_USERNAME", "reviewer")
BASIC_AUTH_PASSWORD = os.environ.get("BASIC_AUTH_PASSWORD", "")

_basic = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials | None = Depends(_basic)) -> None:
    """Gate every route on HTTP Basic auth.

    If BASIC_AUTH_PASSWORD is empty the gate is open, which keeps `make serve-mock`
    and the test suite frictionless on a laptop. Anywhere the app is actually
    reachable, set the variable. /api/health stays open regardless so a platform
    health check can reach it without credentials.

    Both the username and the password are compared with `secrets.compare_digest`,
    which takes the same time whether the first character is wrong or the last. A
    plain `==` returns faster on an early mismatch, and that timing difference can be
    measured over enough requests to recover the password one character at a time.
    """
    if not BASIC_AUTH_PASSWORD:
        return
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = secrets.compare_digest(credentials.username, BASIC_AUTH_USERNAME)
    pass_ok = secrets.compare_digest(credentials.password, BASIC_AUTH_PASSWORD)
    # Both comparisons always run, so the reply takes the same time whether the
    # username was wrong, the password was wrong, or both.
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


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


@app.get("/api/catalog", dependencies=[Depends(require_auth)])
async def get_catalog() -> dict[str, Any]:
    return catalog.catalog_payload()


@app.get("/api/corpus", dependencies=[Depends(require_auth)])
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


@app.get("/api/prompt", dependencies=[Depends(require_auth)])
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


@app.get("/api/runs", dependencies=[Depends(require_auth)])
async def get_runs() -> dict[str, Any]:
    return {"runs": store.list_runs(settings.runs_dir)}


@app.get("/api/runs/{filename}", dependencies=[Depends(require_auth)])
async def get_run(filename: str) -> Any:
    try:
        return store.load_run(settings.runs_dir, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.get("/api/progress", dependencies=[Depends(require_auth)])
async def get_progress() -> dict[str, Any]:
    if _current is None:
        return {"state": "idle"}
    return _current.payload()


@app.get("/api/result", dependencies=[Depends(require_auth)])
async def get_result() -> Any:
    if _last_result is None:
        raise HTTPException(status_code=404, detail="no run has completed in this process yet")
    return _last_result


@app.post("/api/run", dependencies=[Depends(require_auth)])
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


@app.get("/", dependencies=[Depends(require_auth)])
async def index() -> Any:
    idx = STATIC_DIR / "index.html"
    if not idx.is_file():
        return JSONResponse({"detail": "UI not built"}, status_code=500)
    return FileResponse(str(idx))
