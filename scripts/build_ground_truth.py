#!/usr/bin/env python3
"""
Build the answer key from the saved issues.

Two sources, applied in this order.

First, the maintainers' own labels, translated into the six categories through a
lookup table. Free, covers about 300 issues, and lopsided and inconsistent in ways
ANNOTATION_GUIDE.md goes into.

Second, the labels I wrote myself (data/ground_truth/hand_labels.json). A spread of
the issues with no useful maintainer label, weighted towards the categories the
first source can't supply. These win where the two overlap.

Then three more passes:

  - Issues whose labels point at more than one category get settled by a written
    priority order, or dropped if that doesn't resolve it.
  - The bot-written CVE issues get flagged, so the security category can be
    reported with and without them.
  - The whole thing is split into dev and test, balanced by category with a fixed
    seed. Prompt work only ever happens against dev, and the headline numbers come
    from test. Without that split, tuning a prompt until the numbers look good is
    just fitting the test.

Usage:
    python scripts/build_ground_truth.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "corpus" / "doctl-issues-snapshot.json"
HAND = ROOT / "data" / "ground_truth" / "hand_labels.json"
OUT = ROOT / "data" / "ground_truth" / "gold.json"

SCHEMA = ["bug", "enhancement", "question", "documentation", "security", "other"]

# doctl's labels, translated into the six categories. Only labels that say
# something about *category* are in here. The rest (hacktoberfest, good first
# issue, waiting-response, snap, windows, do-api, api-parity, wip, blocked, and so
# on) describe process or component, not category, so they're skipped.
LABEL_MAP = {
    "bug": "bug",
    "suggestion": "enhancement",      # doctl's historical term for feature requests
    "enhancement": "enhancement",     # the newer term; both are in active use
    "security vulnerability": "security",
    "question": "question",
    "troubleshooting": "question",
    "docs": "documentation",
    "duplicate": "other",
}

# Priority order for issues whose labels point at several categories at once.
# 'security' beats everything, because sending a vulnerability to the wrong place is
# the expensive mistake. 'duplicate' (which lands in 'other') beats the topic,
# because a duplicate bug still gets handled as a duplicate. 'docs' beats
# 'suggestion', because a doc request tagged as a suggestion is still a doc change.
CONFLICT_PRECEDENCE = ["security", "other", "documentation", "bug", "enhancement", "question"]

# Output from the Mend/WhiteSource dependency scanner. Matched on the shape of the
# title rather than the bot's username, because the account has changed over the years.
TEMPLATED_RE = re.compile(r"CVE-\d{4}-\d+.*detected in", re.IGNORECASE)

DEV_FRACTION = 0.30
SPLIT_SEED = 20260811


def main() -> int:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    hand = json.loads(HAND.read_text(encoding="utf-8"))

    overlay: dict[int, tuple[str, str]] = {}
    for label, entries in hand["labels"].items():
        if label not in SCHEMA:
            raise SystemExit(f"hand_labels.json uses unknown label {label!r}")
        for num, reason in entries.items():
            overlay[int(num)] = (label, reason)
    excluded_by_hand = {int(k): v for k, v in hand["excluded"].items()}

    gold: list[dict[str, Any]] = []
    stats = {
        "tier_maintainer": 0,
        "tier_hand": 0,
        "conflicts_resolved": 0,
        "conflicts_dropped": 0,
        "excluded_ambiguous": 0,
        "unlabeled_left_unscored": 0,
        "overlay_vs_maintainer_disagreements": [],
    }

    for issue in snapshot["issues"]:
        num = issue["number"]
        raw = issue["maintainer_labels"]
        mapped = sorted({LABEL_MAP[l] for l in raw if l in LABEL_MAP})
        templated = bool(TEMPLATED_RE.search(issue["title"]))

        if num in excluded_by_hand:
            stats["excluded_ambiguous"] += 1
            continue

        label: str | None = None
        source = ""
        note = ""

        if num in overlay:
            label, note = overlay[num]
            source = "hand"
            stats["tier_hand"] += 1
            # If the maintainers had a view too, note it when we disagree.
            if len(mapped) == 1 and mapped[0] != label:
                stats["overlay_vs_maintainer_disagreements"].append(
                    {"number": num, "maintainer": mapped[0], "hand": label}
                )
        elif len(mapped) == 1:
            label = mapped[0]
            source = "maintainer"
            note = f"maintainer labels {raw}"
            stats["tier_maintainer"] += 1
        elif len(mapped) > 1:
            for candidate in CONFLICT_PRECEDENCE:
                if candidate in mapped:
                    label = candidate
                    break
            if label:
                source = "maintainer_conflict_resolved"
                note = f"labels mapped to {mapped}; precedence chose {label}"
                stats["conflicts_resolved"] += 1
                stats["tier_maintainer"] += 1
            else:
                stats["conflicts_dropped"] += 1
                continue
        else:
            stats["unlabeled_left_unscored"] += 1
            continue

        gold.append(
            {
                "number": num,
                "label": label,
                "label_source": source,
                "rationale": note,
                "templated": templated,
                "maintainer_labels": raw,
            }
        )

    # Split into dev and test, keeping the category mix even in both. That balance
    # matters because `other` only has a handful of members, and a plain random
    # split could drop all of them on one side and leave the other unmeasurable.
    rng = random.Random(SPLIT_SEED)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold:
        by_label[row["label"]].append(row)

    for label in sorted(by_label):
        rows = sorted(by_label[label], key=lambda r: r["number"])
        rng.shuffle(rows)
        n_dev = max(1, round(len(rows) * DEV_FRACTION)) if len(rows) > 2 else 0
        for idx, row in enumerate(rows):
            row["split"] = "dev" if idx < n_dev else "test"

    gold.sort(key=lambda r: r["number"])

    dist = Counter(r["label"] for r in gold)
    dist_dev = Counter(r["label"] for r in gold if r["split"] == "dev")
    dist_test = Counter(r["label"] for r in gold if r["split"] == "test")
    templated_ct = Counter(r["label"] for r in gold if r["templated"])

    out = {
        "schema_version": 1,
        "corpus_hash": snapshot["corpus_hash"],
        "schema": SCHEMA,
        "label_map": LABEL_MAP,
        "conflict_precedence": CONFLICT_PRECEDENCE,
        "split": {"dev_fraction": DEV_FRACTION, "seed": SPLIT_SEED},
        "stats": stats,
        "distribution": {
            "all": dict(dist),
            "dev": dict(dist_dev),
            "test": dict(dist_test),
            "templated": dict(templated_ct),
        },
        "totals": {
            "corpus": len(snapshot["issues"]),
            "gold": len(gold),
            "dev": sum(dist_dev.values()),
            "test": sum(dist_test.values()),
            "unscored": len(snapshot["issues"]) - len(gold),
        },
        "items": gold,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
