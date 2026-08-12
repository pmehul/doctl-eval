#!/usr/bin/env python3
"""
Tests every candidate model. This is the step that picks the two the UI compares.

The exercise is clear that the two recommended models have to come *out of* testing
more than two, not be assumed at the start. So this runs all of them over the same
issues and prints a ranked table.

Three things about how it's set up.

It runs on the dev half, not the test half. Picking the winner on test and then
quoting its test score flatters it: with 11 candidates you'd be reporting the best of
11 noisy attempts as though it were a fair estimate. Dev picks the winner. Test
confirms it once, in the app.

Same concurrency budget, same prompt, same issues for everyone. The model name is the
only thing that changes.

Ranked on macro-F1 first, then on cost per correct answer. Accuracy on its own would
reward a model that just says "bug" a lot, and cost per call on its own would reward
one that's cheap and wrong.

Usage:
    # everything, on dev
    python scripts/screen_models.py --split dev --concurrency 8

    # just a couple, for a quick check
    python scripts/screen_models.py --models openai-gpt-oss-20b,openai-gpt-oss-120b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import catalog, metrics                      # noqa: E402
from app.config import Settings                       # noqa: E402
from app.corpus import Issue, load_corpus             # noqa: E402
from app.inference import make_client                 # noqa: E402
from app.prompting import PROMPT_VERSION              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "screening"


async def screen_one(
    settings: Settings, model_id: str, issues: list[Issue], concurrency: int
) -> dict:
    client = make_client(settings, concurrency)
    gate = asyncio.Semaphore(concurrency)
    rows: list[dict] = []

    # Progress is printed as calls land, not just when the model finishes.
    #
    # Screening used to print the model name and then nothing until every issue was
    # done. That reads as a hang the moment calls get slow: raising the output cap
    # for a thinking model took one model from four minutes to roughly forty, with
    # no output in between, and the only reasonable conclusion from the outside was
    # that it had wedged. It also hid the failure type until the very end, which is
    # the thing you most want to see early.
    done = 0
    errs: dict[str, int] = {}

    async def one(issue: Issue) -> None:
        nonlocal done
        async with gate:
            res = await client.classify(issue, model_id)
        row = asdict(res)
        row["gold_label"] = issue.gold_label
        row["templated"] = issue.templated
        rows.append(row)

        done += 1
        if res.error_type:
            errs[res.error_type] = errs.get(res.error_type, 0) + 1
        if done % 10 == 0 or done == len(issues):
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            eta = (len(issues) - done) / rate if rate else 0.0
            detail = (
                "  " + ", ".join(f"{k} {v}" for k, v in sorted(errs.items(), key=lambda kv: -kv[1]))
                if errs
                else ""
            )
            print(
                f"    {done}/{len(issues)}  {rate:.1f}/s  "
                f"elapsed {elapsed:.0f}s  eta {eta:.0f}s{detail}",
                flush=True,
            )

    started = time.perf_counter()
    try:
        await asyncio.gather(*(one(i) for i in issues))
    finally:
        await client.aclose()
    wall = time.perf_counter() - started

    quality = metrics.score(rows)
    ops = metrics.operational(rows, wall, concurrency)
    spec = catalog.get(model_id)

    # Whether a model sticks to the format is counted apart from whether it picks
    # the right label. A model that only ever gets through on the last-resort scan
    # is ignoring instructions, which is a risk in production even if it scores well.
    strategies: dict[str, int] = {}
    for r in rows:
        s = r.get("parse_strategy")
        if s:
            strategies[s] = strategies.get(s, 0) + 1

    return {
        "model_id": model_id,
        "label": spec.label,
        "params": spec.params,
        "architecture": spec.architecture,
        "reasoning": spec.reasoning,
        "usd_per_m_input": spec.usd_per_m_input,
        "usd_per_m_output": spec.usd_per_m_output,
        "quality": quality,
        "operational": ops,
        "parse_strategies": strategies,
    }


def table(results: list[dict], wall_total: float) -> str:
    """Markdown table, ready to paste into the README."""
    header = (
        "| model | params | arch | macro-F1 | macro-F1 excl. templated | accuracy | "
        "p50 ms | p95 ms | mean out tok | $/call | $/correct | err % |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in results:
        q, o = r["quality"], r["operational"]
        per_correct = o["cost"]["per_correct_usd"]
        lines.append(
            "| `{id}` | {params} | {arch}{reason} | {mf1:.3f} | {mf1x:.3f} | {acc:.1%} | "
            "{p50} | {p95} | {out:.0f} | {pc} | {pcorr} | {err:.1%} |".format(
                id=r["model_id"],
                params=r["params"],
                arch=r["architecture"],
                reason=" · reasoning" if r["reasoning"] else "",
                mf1=q["macro_f1"],
                mf1x=q["macro_f1_excl_templated"],
                acc=q["accuracy"],
                p50=f"{o['latency_ms']['p50']:.0f}" if o["latency_ms"]["p50"] else "—",
                p95=f"{o['latency_ms']['p95']:.0f}" if o["latency_ms"]["p95"] else "—",
                out=o["tokens"]["mean_completion"],
                pc=f"${o['cost']['per_call_usd']:.3g}",
                pcorr=f"${per_correct:.3g}" if per_correct else "—",
                err=o["error_rate"],
            )
        )
    return header + "\n".join(lines)


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="gold split to score on. Selection should use dev.")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--models", default=None, help="comma-separated subset of model ids")
    args = ap.parse_args()

    settings = Settings()
    problems = settings.validate()
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 2

    concurrency = args.concurrency or settings.concurrency
    corpus = load_corpus()
    issues = list(corpus.scored(args.split))
    if not issues:
        print(f"no issues in split {args.split!r}", file=sys.stderr)
        return 2

    model_ids = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else [m.id for m in catalog.CATALOG]
    )
    for mid in model_ids:
        catalog.get(mid)  # fail fast on typos

    if args.split == "test":
        print(
            "warning: screening on the test split. Selecting the winner here and then\n"
            "         quoting its test score is selection bias. Use --split dev.\n",
            file=sys.stderr,
        )

    print(
        f"screening {len(model_ids)} models on {len(issues)} {args.split}-split issues "
        f"at concurrency {concurrency} (provider={settings.provider})\n",
        file=sys.stderr,
    )

    results: list[dict] = []
    wall_start = time.perf_counter()
    for mid in model_ids:
        # Newline, not end="": progress lines now print underneath this header, so
        # holding the line open left the summary dangling off the last of them.
        print(f"  {mid} ...", flush=True, file=sys.stderr)
        r = await screen_one(settings, mid, issues, concurrency)
        results.append(r)
        print(
            f"  -> macro-F1 {r['quality']['macro_f1']:.3f}"
            f"  acc {r['quality']['accuracy']:.1%}"
            f"  p95 {r['operational']['latency_ms']['p95'] or 0:.0f}ms"
            f"  ${r['operational']['cost']['per_call_usd']:.3g}/call"
            f"  err {r['operational']['error_rate']:.1%}",
            file=sys.stderr,
        )
    wall_total = time.perf_counter() - wall_start

    # Sorted on quality first, then on what a correct answer costs. Spelled out here
    # so you can disagree with the order instead of working it out from the output.
    results.sort(
        key=lambda r: (
            -r["quality"]["macro_f1"],
            r["operational"]["cost"]["per_correct_usd"] or float("inf"),
        )
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": settings.provider,
        "simulated": settings.is_mock,
        "split": args.split,
        "n_issues": len(issues),
        "concurrency": concurrency,
        "prompt_version": PROMPT_VERSION,
        "corpus_hash": corpus.corpus_hash,
        "pricing_source": catalog.PRICING_SOURCE,
        "pricing_verified": catalog.PRICING_VERIFIED,
        "wall_clock_s": wall_total,
        "results": results,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "sim" if settings.is_mock else "live"
    stem = f"screening-{tag}-{args.split}-{corpus.corpus_hash}"
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = table(results, wall_total)
    (OUT_DIR / f"{stem}.md").write_text(
        f"# Model screening — {args.split} split, {len(issues)} issues\n\n"
        f"- provider: `{settings.provider}`"
        + ("  **(SIMULATED — not evidence)**" if settings.is_mock else "")
        + f"\n- corpus: `{corpus.corpus_hash}`\n"
        f"- prompt: `{PROMPT_VERSION}`\n"
        f"- concurrency: {concurrency}\n"
        f"- rates: {catalog.PRICING_SOURCE} (verified {catalog.PRICING_VERIFIED})\n\n"
        + md
        + "\n",
        encoding="utf-8",
    )

    print("\n" + md)
    print(f"\nwrote {(OUT_DIR / f'{stem}.json').relative_to(ROOT)}")
    print(f"wrote {(OUT_DIR / f'{stem}.md').relative_to(ROOT)}")
    if settings.is_mock:
        print(
            "\nNOTE: PROVIDER=mock. These are simulated numbers and cannot support a "
            "recommendation.\n      Re-run with PROVIDER=digitalocean and a real API key.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
