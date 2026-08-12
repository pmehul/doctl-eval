"""Scoring, plus the speed and cost figures.

Written out by hand instead of importing scikit-learn. Six-class precision and
recall is about twenty lines, and the exercise asks for the maths to be checkable.
You can read the F1 definition below and confirm it. You can't do that with an
import, and it saves the reviewer building a large dependency to run this.

Four things here are worth knowing about.

macro-F1 is the headline number, not accuracy. The gold set holds 173 bug, 112
enhancement, 28 question, 26 security, 18 documentation, 5 other. A model that
falls back to "bug" whenever it's unsure will look fine on accuracy and be
useless on the four small buckets. Macro-F1 gives every bucket equal weight, so
that trick stops working.

macro-F1 is also reported with the bot-written security issues taken out. All 26
of them are scanner reports shaped like "CVE-XXXX-YYYY detected in <package>",
which a regex can spot. Leaving them in makes every model look better on one
bucket out of six, so both numbers are shown and you can see the size of it.

Percentiles pick the nearest actual measurement instead of averaging between two.
With a few hundred requests the difference is tiny, and this way p95 is always a
latency that really happened, which is what you want when someone asks which
request that was.

Latency is timed on this side, start to finish per request. That's what the
caller waits. Timing only the model's own generation would look better and answer
a question nobody asked.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .corpus import LABELS

TEMPLATED_CLASS = "security"


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile. q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil(q / 100 * len(ordered))
    idx = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[idx]


def confusion_matrix(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """rows = ground truth, cols = prediction."""
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS}
    for truth, pred in pairs:
        if truth in matrix and pred in matrix[truth]:
            matrix[truth][pred] += 1
    return matrix


def per_class(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        fp = sum(matrix[t][label] for t in LABELS if t != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {
            "support": support,
            "predicted": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out


def _macro_f1(stats: dict[str, dict[str, float | int]], skip: set[str]) -> float:
    # Classes with zero support are excluded from the average. Including them
    # would contribute a hard 0.0 for a class that was never in the gold set,
    # which drags the headline number for a reason unrelated to model quality.
    present = [
        float(s["f1"]) for label, s in stats.items() if label not in skip and int(s["support"]) > 0
    ]
    return sum(present) / len(present) if present else 0.0


def score(results: Sequence[dict]) -> dict:
    """Score the issues we know the answer to.

    Each row needs `gold_label`, `predicted_label` and `templated`. Rows where the
    call failed have no prediction, so they can't count towards accuracy, but the
    count is reported. A model that drops 10% of calls and gets the rest right is
    not the more accurate model, and quietly dropping failures is how that gets
    hidden.
    """
    usable = [r for r in results if r.get("predicted_label")]
    failed = len(results) - len(usable)

    pairs = [(r["gold_label"], r["predicted_label"]) for r in usable]
    matrix = confusion_matrix(pairs)
    stats = per_class(matrix)

    correct = sum(1 for t, p in pairs if t == p)
    accuracy = correct / len(pairs) if pairs else 0.0

    # Same numbers with the bot-generated CVE reports removed.
    non_templated = [r for r in usable if not r.get("templated")]
    nt_pairs = [(r["gold_label"], r["predicted_label"]) for r in non_templated]
    nt_matrix = confusion_matrix(nt_pairs)
    nt_stats = per_class(nt_matrix)
    nt_correct = sum(1 for t, p in nt_pairs if t == p)

    return {
        "n_scored": len(results),
        "n_usable": len(usable),
        "n_failed": failed,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": _macro_f1(stats, skip=set()),
        "macro_f1_excl_templated": _macro_f1(nt_stats, skip=set()),
        "accuracy_excl_templated": (nt_correct / len(nt_pairs)) if nt_pairs else 0.0,
        "n_excl_templated": len(nt_pairs),
        "per_class": stats,
        "confusion_matrix": matrix,
    }


def operational(results: Sequence[dict], wall_clock_s: float, concurrency: int) -> dict:
    """Cost, speed, throughput and error counts for one model's run.

    The percentiles carry the concurrency they were measured at, because a p95 on
    its own is not something you can act on. The same model at 4 and at 64 gives
    very different numbers, and the second one includes queue time the first
    doesn't.
    """
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    successes = [r for r in results if r.get("predicted_label")]

    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in results)
    completion_tokens = sum(r.get("completion_tokens") or 0 for r in results)
    total_cost = sum(r.get("total_cost_usd") or 0.0 for r in results)

    errors: dict[str, int] = {}
    for r in results:
        kind = r.get("error_type")
        if kind:
            errors[kind] = errors.get(kind, 0) + 1

    n = len(results)
    n_ok = len(successes)
    correct = sum(1 for r in successes if r.get("gold_label") and r["gold_label"] == r["predicted_label"])

    retries = sum(r.get("attempts", 1) - 1 for r in results)

    return {
        "concurrency": concurrency,
        "wall_clock_s": wall_clock_s,
        "requests": n,
        "successful_requests": n_ok,
        "failed_requests": n - n_ok,
        "retries": retries,
        "error_rate": (n - n_ok) / n if n else 0.0,
        "errors_by_type": errors,
        "throughput_rps": n / wall_clock_s if wall_clock_s > 0 else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "n": len(latencies),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "mean_prompt": prompt_tokens / n if n else 0.0,
            "mean_completion": completion_tokens / n if n else 0.0,
        },
        "cost": {
            "total_usd": total_cost,
            "per_call_usd": total_cost / n if n else 0.0,
            # The headline unit-economics number. Cost per *correct* classification
            # is what a customer actually buys: a model that is 20% cheaper per
            # call but wrong twice as often costs more per useful output. Only
            # defined where ground truth exists.
            "per_correct_usd": (total_cost / correct) if correct else None,
            "correct": correct,
        },
    }


def agreement(rows: Sequence[dict]) -> dict:
    """How often the two models say the same thing, where we have no answer key.

    This is the headline for the unscored view, and it is a stand-in, not a
    quality score. Both models can agree and both be wrong, and the easy buckets
    push the number up. What it is actually good for is sizing the second-opinion
    tier: the issues they disagree on are the ones you'd send to a bigger model or
    a person, so the size of that set feeds straight into what production costs.
    """
    both = [r for r in rows if r.get("a_label") and r.get("b_label")]
    agreed = sum(1 for r in both if r["a_label"] == r["b_label"])
    per_class_a: dict[str, int] = {l: 0 for l in LABELS}
    per_class_b: dict[str, int] = {l: 0 for l in LABELS}
    for r in rows:
        if r.get("a_label") in per_class_a:
            per_class_a[r["a_label"]] += 1
        if r.get("b_label") in per_class_b:
            per_class_b[r["b_label"]] += 1

    # Which pairs they disagree on most. This is the useful part: a big
    # documentation/enhancement count tells you exactly which boundary the prompt
    # or the category list is failing to pin down.
    pairs: dict[str, int] = {}
    for r in both:
        if r["a_label"] != r["b_label"]:
            key = f"{r['a_label']} vs {r['b_label']}"
            pairs[key] = pairs.get(key, 0) + 1

    return {
        "n": len(rows),
        "n_both_succeeded": len(both),
        "n_agreed": agreed,
        "n_disagreed": len(both) - agreed,
        "agreement_rate": agreed / len(both) if both else 0.0,
        "distribution_a": per_class_a,
        "distribution_b": per_class_b,
        "top_disagreement_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
    }
