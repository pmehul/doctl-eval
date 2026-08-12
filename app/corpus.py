"""Loads the issues and the answer key.

Everything comes off the saved file on disk. Nothing here calls GitHub. That's what
makes "the same issues every run" a fact instead of a promise: there is no route by
which one run could see different issues from the last, and the fingerprint on every
result says which set was scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import settings

LABELS = ("bug", "enhancement", "question", "documentation", "security", "other")


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    state: str
    html_url: str
    maintainer_labels: tuple[str, ...]
    gold_label: str | None
    gold_source: str | None
    gold_split: str | None
    templated: bool

    @property
    def scored(self) -> bool:
        return self.gold_label is not None


@dataclass(frozen=True)
class Corpus:
    """The saved issues, plus the subset a run will actually sort.

    `issues` is always the full set, and it's what `by_number` looks in. `workload`
    is what a run classifies, which MAX_ISSUES can shrink for a quick test.

    Splitting these two isn't fussiness. MAX_ISSUES used to cut down `issues`
    itself, which quietly removed the worked-example issues from the lookup and
    changed the prompt. A setting whose only job is "go faster" must not change the
    experiment. MAX_ISSUES caps how much work a run does and says nothing about
    what the dataset holds.
    """

    repo: str
    corpus_hash: str
    frozen_at: str
    issues: tuple[Issue, ...]
    workload: tuple[Issue, ...]
    gold_stats: dict

    def scored(self, split: str) -> tuple[Issue, ...]:
        if split == "all":
            return tuple(i for i in self.workload if i.scored)
        return tuple(i for i in self.workload if i.scored and i.gold_split == split)

    def unscored(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.workload if not i.scored)

    def by_number(self, number: int) -> Issue | None:
        return next((i for i in self.issues if i.number == number), None)


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path}.\n"
            "Run:  python scripts/ingest_issues.py && python scripts/build_ground_truth.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    snap = _read(settings.corpus_path)
    gold = _read(settings.gold_path)

    if gold.get("corpus_hash") != snap.get("corpus_hash"):
        raise SystemExit(
            "corpus/gold hash mismatch: the gold set was built against a different "
            f"snapshot (gold={gold.get('corpus_hash')} snapshot={snap.get('corpus_hash')}).\n"
            "Rebuild with: python scripts/build_ground_truth.py"
        )

    gold_by_num = {item["number"]: item for item in gold["items"]}

    issues: list[Issue] = []
    for raw in snap["issues"]:
        g = gold_by_num.get(raw["number"])
        issues.append(
            Issue(
                number=raw["number"],
                title=raw["title"],
                body=raw["body"],
                state=raw["state"],
                html_url=raw["html_url"],
                maintainer_labels=tuple(raw["maintainer_labels"]),
                gold_label=g["label"] if g else None,
                gold_source=g["label_source"] if g else None,
                gold_split=g["split"] if g else None,
                templated=bool(g["templated"]) if g else False,
            )
        )

    issues.sort(key=lambda i: i.number)

    # Shrinking for a quick test. It takes an even spread rather than the first N,
    # so a smaller run still covers the repo's whole ten-year history. The first N
    # would all be from 2015, and the writing style and labelling habits back then
    # don't look like the rest of the set.
    workload = issues
    if settings.max_issues > 0 and settings.max_issues < len(issues):
        stride = len(issues) / settings.max_issues
        workload = [issues[int(k * stride)] for k in range(settings.max_issues)]

    return Corpus(
        repo=snap["repo"],
        corpus_hash=snap["corpus_hash"],
        frozen_at=snap["frozen_at"],
        issues=tuple(issues),
        workload=tuple(workload),
        gold_stats={
            "distribution": gold["distribution"],
            "totals": gold["totals"],
            "stats": gold["stats"],
            "split": gold["split"],
        },
    )
