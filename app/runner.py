"""Runs the comparison.

One unit of work is one issue on one model, sent as one request. Nothing in this
file batches anything.

There's a single concurrency limit for the whole run, shared by both models, not one
each. If each model had its own budget of N they'd be running N+N together, and
neither model's p95 would match the concurrency printed beside it. One shared limit
means the number on screen is the number that produced the latency.

Both models go over the same issues in the same run, alternating, instead of one
after the other. If the provider slows down mid-run it hits both roughly equally
rather than punishing whichever went second, and the total wall clock is the one a
customer would actually see when comparing two models.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

from . import catalog, metrics
from .config import Settings
from .corpus import Issue, load_corpus
from .inference import CallResult, make_client
from .prompting import PROMPT_VERSION

ProgressFn = Callable[[dict], None]


class RunProgress:
    """Mutable snapshot of an in-flight run, polled by the UI."""

    def __init__(self, run_id: str, total: int) -> None:
        self.run_id = run_id
        self.total = total
        self.completed = 0
        self.errors = 0
        # Errors are counted per type, not just totalled. A bare "264 errors" says
        # something is wrong without saying what, and each type points at a
        # different fix: rate_limit means lower the concurrency, timeout means
        # raise the deadline, bad_json means repair the prompt, auth means the key
        # is wrong. Naming the type live turns a stuck run into a diagnosis.
        self.errors_by_type: dict[str, int] = {}
        self.started_at = time.time()
        self.state = "running"
        self.message = ""

    def payload(self) -> dict:
        elapsed = time.time() - self.started_at
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.completed
        return {
            "run_id": self.run_id,
            "state": self.state,
            "total": self.total,
            "completed": self.completed,
            "errors": self.errors,
            "errors_by_type": dict(
                sorted(self.errors_by_type.items(), key=lambda kv: -kv[1])
            ),
            "elapsed_s": elapsed,
            "throughput_rps": rate,
            "eta_s": (remaining / rate) if rate > 0 and remaining > 0 else 0.0,
            "message": self.message,
        }


async def execute_run(
    settings: Settings,
    model_a: str,
    model_b: str,
    concurrency: int,
    scored_split: str,
    progress: RunProgress | None = None,
) -> dict:
    corpus = load_corpus()
    spec_a, spec_b = catalog.get(model_a), catalog.get(model_b)

    issues = list(corpus.workload)
    # Only score the split that was asked for. Everything else still gets
    # classified, it just lands in the unscored view. A customer's real pile of
    # issues is mostly unlabelled, so that path isn't an afterthought here.
    scored_numbers = {i.number for i in corpus.scored(scored_split)}

    tasks: list[tuple[Issue, str]] = []
    for issue in issues:
        tasks.append((issue, model_a))
        tasks.append((issue, model_b))

    client = make_client(settings, concurrency)
    gate = asyncio.Semaphore(concurrency)
    results: dict[tuple[int, str], CallResult] = {}

    async def one(issue: Issue, model_id: str) -> None:
        async with gate:
            res = await client.classify(issue, model_id)
        results[(issue.number, model_id)] = res
        if progress is not None:
            progress.completed += 1
            if res.error_type:
                progress.errors += 1
                progress.errors_by_type[res.error_type] = (
                    progress.errors_by_type.get(res.error_type, 0) + 1
                )

    wall_start = time.perf_counter()
    try:
        await asyncio.gather(*(one(i, m) for i, m in tasks))
    finally:
        await client.aclose()
    wall_clock_s = time.perf_counter() - wall_start

    # The wall clock is taken once, for the whole run. Giving each model its own
    # would be made up: they shared one concurrency budget and one window.
    per_model_rows: dict[str, list[dict]] = {model_a: [], model_b: []}
    scored_rows: dict[str, list[dict]] = {model_a: [], model_b: []}

    for issue in issues:
        for model_id in (model_a, model_b):
            res = results[(issue.number, model_id)]
            row = asdict(res)
            row["gold_label"] = issue.gold_label
            row["templated"] = issue.templated
            row["is_scored"] = issue.number in scored_numbers
            per_model_rows[model_id].append(row)
            if issue.number in scored_numbers:
                scored_rows[model_id].append(row)

    # --- scored view ---------------------------------------------------
    scored_items = []
    for issue in corpus.scored(scored_split):
        ra = results[(issue.number, model_a)]
        rb = results[(issue.number, model_b)]
        scored_items.append(
            {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "html_url": issue.html_url,
                "state": issue.state,
                "gold_label": issue.gold_label,
                "gold_source": issue.gold_source,
                "templated": issue.templated,
                "maintainer_labels": list(issue.maintainer_labels),
                "a": _item_view(ra),
                "b": _item_view(rb),
                "a_correct": ra.predicted_label == issue.gold_label,
                "b_correct": rb.predicted_label == issue.gold_label,
                "models_disagree": ra.predicted_label != rb.predicted_label,
            }
        )

    # --- unscored view -------------------------------------------------
    # Unscored means "not scored in this run", not "has no gold label".
    #
    # This iterated corpus.unscored(), which is the 174 issues with no label at all.
    # With SCORED_SPLIT=test the scored view holds 253, so the two views showed 427
    # of the 536 issues that were classified and paid for. The 109 dev-split issues
    # have labels but are deliberately held out of scoring, so they appeared in
    # neither view and simply vanished from the UI.
    #
    # The exercise asks for a scored view for the labeled subset and an unscored view
    # for the rest, and the rest of 536 after 253 is 283. Partitioning on what this
    # run actually scored makes the two views add up, and it gives the agreement rate
    # a wider sample: 283 issues instead of 174.
    unscored_items = []
    agreement_rows = []
    for issue in (i for i in issues if i.number not in scored_numbers):
        ra = results[(issue.number, model_a)]
        rb = results[(issue.number, model_b)]
        unscored_items.append(
            {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "html_url": issue.html_url,
                "state": issue.state,
                "maintainer_labels": list(issue.maintainer_labels),
                "a": _item_view(ra),
                "b": _item_view(rb),
                "models_disagree": ra.predicted_label != rb.predicted_label,
            }
        )
        agreement_rows.append({"a_label": ra.predicted_label, "b_label": rb.predicted_label})

    payload = {
        "run_id": progress.run_id if progress else uuid.uuid4().hex[:12],
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "provider": settings.provider,
        "simulated": settings.is_mock,
        "corpus": {
            "repo": corpus.repo,
            "corpus_hash": corpus.corpus_hash,
            "frozen_at": corpus.frozen_at,
            "n_issues": len(issues),
            "n_scored": len(scored_numbers),
            "n_unscored": len(issues) - len(scored_numbers),
            "scored_split": scored_split,
        },
        "config": {
            "concurrency": concurrency,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "reasoning_max_tokens": settings.reasoning_max_tokens,
            "request_timeout_s": settings.request_timeout_s,
            "max_retries": settings.max_retries,
            "prompt_version": PROMPT_VERSION,
            "mock_time_scale": settings.mock_time_scale if settings.is_mock else None,
        },
        "models": {
            "a": {**asdict(spec_a), "slot": "A"},
            "b": {**asdict(spec_b), "slot": "B"},
        },
        "pricing": {
            "source": catalog.PRICING_SOURCE,
            "verified": catalog.PRICING_VERIFIED,
        },
        "scored": {
            "split": scored_split,
            "a": metrics.score(scored_rows[model_a]),
            "b": metrics.score(scored_rows[model_b]),
            "items": scored_items,
        },
        "unscored": {
            "agreement": metrics.agreement(agreement_rows),
            "items": unscored_items,
        },
        "operational": {
            "wall_clock_s": wall_clock_s,
            "total_requests": len(tasks),
            # Throughput for the run as a whole: both models' requests over the one
            # shared budget. The per-model figures below are each model's own count
            # over the same window, so they add up to this. Showing only the
            # per-model number would imply each had the full budget to itself.
            "aggregate_throughput_rps": len(tasks) / wall_clock_s if wall_clock_s > 0 else 0.0,
            "a": metrics.operational(per_model_rows[model_a], wall_clock_s, concurrency),
            "b": metrics.operational(per_model_rows[model_b], wall_clock_s, concurrency),
        },
    }
    return payload


def _item_view(res: CallResult) -> dict:
    """What the UI gets for one issue, cost breakdown included.

    The cost fields ride along on every item, not just the totals, so you can open
    any single issue and check that
    prompt_tokens/1e6*rate_in + completion_tokens/1e6*rate_out
    matches the figure on screen. That's the "show your working" requirement, met
    at the level of one call.
    """
    return {
        "label": res.predicted_label,
        "confidence": res.confidence,
        "parse_strategy": res.parse_strategy,
        "raw_output": res.raw_output,
        "latency_ms": res.latency_ms,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "usd_per_m_input": res.usd_per_m_input,
        "usd_per_m_output": res.usd_per_m_output,
        "input_cost_usd": res.input_cost_usd,
        "output_cost_usd": res.output_cost_usd,
        "total_cost_usd": res.total_cost_usd,
        "error_type": res.error_type,
        "error_detail": res.error_detail,
        "attempts": res.attempts,
        "http_status": res.http_status,
        "simulated": res.simulated,
    }
