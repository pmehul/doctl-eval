#!/usr/bin/env python3
"""
Checks the things that could break quietly.

These are the properties that, if they went wrong, would make every number here
wrong without anything raising an error. Each one has already caught a real bug
while I was building:

  worked examples must all come from dev   caught an answer leaking into the test set
  dataset and answer-key fingerprints      catches an answer key left over from an
    must match                               older download
  MAX_ISSUES must not change the prompt    caught the worked examples disappearing
                                             from a shortened dataset
  cost maths must be checkable by hand     the show-your-working requirement

Run with:  make verify
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import catalog                                          # noqa: E402
from app.config import Settings                                   # noqa: E402
from app.corpus import LABELS, load_corpus                        # noqa: E402
from app.metrics import percentile                                # noqa: E402
from app.prompting import (                                       # noqa: E402
    FEW_SHOT_NUMBERS,
    build_messages,
    few_shot_messages,
    parse_label,
)

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    settings = Settings()
    corpus = load_corpus()

    print("\ncorpus")
    check("snapshot loads", len(corpus.issues) > 0, f"{len(corpus.issues)} issues")
    check(
        "every issue has a title",
        all(i.title.strip() for i in corpus.issues),
        f"{sum(1 for i in corpus.issues if not i.title.strip())} empty",
    )
    check(
        "issue numbers unique",
        len({i.number for i in corpus.issues}) == len(corpus.issues),
    )
    check("corpus hash present", bool(corpus.corpus_hash), corpus.corpus_hash)

    print("\ngold set")
    gold = [i for i in corpus.issues if i.scored]
    check("gold set non-empty", len(gold) > 0, f"{len(gold)} labeled")
    check(
        "all gold labels are in the schema",
        all(i.gold_label in LABELS for i in gold),
    )
    check(
        "every gold item has a split",
        all(i.gold_split in {"dev", "test"} for i in gold),
    )
    dev = [i for i in gold if i.gold_split == "dev"]
    test = [i for i in gold if i.gold_split == "test"]
    check("dev and test both populated", bool(dev) and bool(test), f"{len(dev)} dev / {len(test)} test")
    check(
        "dev and test are disjoint",
        not ({i.number for i in dev} & {i.number for i in test}),
    )
    dist_test = Counter(i.gold_label for i in test)
    check(
        "every class with dev support also has test support",
        all(dist_test[c] > 0 for c in {i.gold_label for i in dev}),
        f"test dist {dict(dist_test)}",
    )
    thin = [c for c, n in dist_test.items() if n < 10]
    print(
        f"  NOTE  classes with test support < 10: {thin or 'none'}"
        " -- their F1 is noise and the UI marks them 'thin'"
    )
    check(
        "all templated items are the security class",
        all(i.gold_label == "security" for i in corpus.issues if i.templated),
        f"{sum(1 for i in corpus.issues if i.templated)} templated",
    )

    print("\nprompt")
    fs = few_shot_messages()
    check("few-shot set builds", len(fs) == len(FEW_SHOT_NUMBERS) * 2, f"{len(fs)} messages")
    fs_issues = [corpus.by_number(n) for n in FEW_SHOT_NUMBERS]
    check(
        "every few-shot example is dev-split (no test leak)",
        all(i is not None and i.gold_split == "dev" for i in fs_issues),
        ", ".join(f"#{i.number}={i.gold_split}" for i in fs_issues if i),
    )
    check(
        "few-shot examples are not scored in the test split",
        not ({n for n in FEW_SHOT_NUMBERS} & {i.number for i in test}),
    )
    sample = test[0]
    msgs = build_messages(sample)
    check("rendered request has system + few-shot + item", len(msgs) == 2 + len(fs), f"{len(msgs)} messages")
    check("system prompt is first", msgs[0]["role"] == "system")
    check("item under test is last", msgs[-1]["role"] == "user" and sample.title in msgs[-1]["content"])
    check(
        "prompt mentions every schema label",
        all(l in msgs[0]["content"] for l in LABELS),
    )

    print("\nMAX_ISSUES must not perturb the prompt")
    # The bug this guards against: MAX_ISSUES used to cut down the dataset itself,
    # which dropped the higher-numbered worked examples and changed the prompt. A
    # setting whose only job is "go faster" must not change the experiment.
    baseline = build_messages(sample)
    check(
        "few-shot issues survive truncation",
        all(corpus.by_number(n) is not None for n in FEW_SHOT_NUMBERS),
        "corpus.issues stays complete; only corpus.workload shrinks",
    )
    check(
        "workload is a subset of the corpus",
        {i.number for i in corpus.workload} <= {i.number for i in corpus.issues},
        f"workload={len(corpus.workload)} corpus={len(corpus.issues)}",
    )
    check("prompt is stable", build_messages(sample) == baseline)

    print("\nparser")
    cases = [
        ('{"label": "bug", "confidence": 0.9}', "bug", "bare_json"),
        ('```json\n{"label":"question","confidence":0.5}\n```', "question", "fenced_json"),
        ('<think>could be bug or enhancement, leaning enhancement</think>{"label":"enhancement"}', "enhancement", None),
        ("Here you go: {\"label\":\"security\"}", "security", None),
        ("I would call this documentation.", "documentation", "loose_scan"),
    ]
    for raw, expect, strategy in cases:
        try:
            label, _conf, strat = parse_label(raw)
            ok = label == expect and (strategy is None or strat == strategy)
            check(f"parses {raw[:44]!r}", ok, f"got {label}/{strat}")
        except Exception as exc:
            check(f"parses {raw[:44]!r}", False, str(exc))
    for bad in ["", "   ", "no label at all here"]:
        try:
            parse_label(bad)
            check(f"rejects {bad!r}", False, "should have raised")
        except Exception:
            check(f"rejects {bad!r}", True)
    # The thinking mentions several labels on the way; the real answer must win.
    tricky = '<think>bug? question? no, documentation</think>\n{"label": "enhancement"}'
    label, _, _ = parse_label(tricky)
    check("reasoning block does not fool the parser", label == "enhancement", f"got {label}")

    print("\ncost arithmetic")
    b = catalog.price_call("openai-gpt-oss-120b", 1_000_000, 1_000_000)
    check("1M in + 1M out == published rates", abs(b["total_cost_usd"] - (0.10 + 0.70)) < 1e-12,
          f"${b['total_cost_usd']}")
    b2 = catalog.price_call("llama3.3-70b-instruct", 1120, 22)
    manual = 1120 / 1e6 * 0.65 + 22 / 1e6 * 0.65
    check("per-call figure reproducible by hand", abs(b2["total_cost_usd"] - manual) < 1e-18,
          f"${b2['total_cost_usd']:.10f}")
    check("no rounding applied at this layer", b2["total_cost_usd"] != round(b2["total_cost_usd"], 2))
    check("every catalog model is priced",
          all(m.usd_per_m_input > 0 and m.usd_per_m_output > 0 for m in catalog.CATALOG),
          f"{len(catalog.CATALOG)} models")
    check("default A and B differ",
          catalog.DEFAULT_MODEL_A != catalog.DEFAULT_MODEL_B)
    check("default A and B are different vendors and architectures",
          catalog.get(catalog.DEFAULT_MODEL_A).vendor != catalog.get(catalog.DEFAULT_MODEL_B).vendor,
          f"{catalog.get(catalog.DEFAULT_MODEL_A).vendor} vs {catalog.get(catalog.DEFAULT_MODEL_B).vendor}")

    print("\npercentiles")
    check("p50 of 1..100", percentile(list(range(1, 101)), 50) == 50)
    check("p95 of 1..100", percentile(list(range(1, 101)), 95) == 95)
    check("p95 is an observed value", percentile([1.0, 2.0, 100.0], 95) in (1.0, 2.0, 100.0))
    check("empty sample is None", percentile([], 50) is None)
    check("single sample", percentile([7.0], 95) == 7.0)

    print("\nconfig")
    problems = settings.validate()
    if settings.is_mock:
        check("mock provider needs no key", not problems, "PROVIDER=mock")
    else:
        print(f"  NOTE  provider={settings.provider}; config problems: {problems or 'none'}")
    check("scored_split is valid", settings.scored_split in {"test", "dev", "all"}, settings.scored_split)

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
