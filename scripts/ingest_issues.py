#!/usr/bin/env python3
"""
Download the doctl issues once and save them to a file that never changes.

Four decisions here, all worth defending.

The set stays the same because it's saved to disk, not because the query is pinned.
GitHub moves under you: issues get relabelled, reopened, edited, deleted. If the app
called GitHub at run time, two runs a week apart would score different issues and
comparing models would mean nothing. So this downloads once, fingerprints what it
got, and the app only ever reads the file.

Pull requests are left out. GitHub's /issues endpoint hands back PRs as if they were
issues, since they share one numbering. PRs are a different kind of writing and would
pollute a test about triaging issues.

The maintainers' own labels are kept exactly as they are. Turning them into the six
categories is a separate step (scripts/build_ground_truth.py), so that mapping can be
argued with and changed without downloading anything again.

Issue bodies are cut to a fixed length here. It keeps prompt costs comparable between
models and stops a handful of 40,000-character bug reports full of logs from eating
the token bill. Which issues were cut is recorded, so it's visible rather than quiet.

Usage:
    python scripts/ingest_issues.py [--token GITHUB_TOKEN] [--max-issues 600]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = "digitalocean/doctl"
API = "https://api.github.com"
PER_PAGE = 100

# How much of an issue body the model sees. 1600 characters is about 400 tokens,
# which comfortably covers the part that decides the answer: the title, the first
# few paragraphs, and the steps to reproduce. Pasted stack traces cost money and
# add nothing.
BODY_CHAR_BUDGET = 1600

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "corpus"


def http_get_json(url: str, token: str | None, retries: int = 5) -> tuple[Any, dict[str, str]]:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "doctl-eval-harness-ingest/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return json.loads(resp.read().decode("utf-8")), headers
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            last_err = exc
            # Primary/secondary rate limiting.
            if exc.code in (403, 429):
                reset = exc.headers.get("x-ratelimit-reset")
                remaining = exc.headers.get("x-ratelimit-remaining")
                if remaining == "0" and reset:
                    wait = max(0, int(reset) - int(time.time())) + 2
                    print(
                        f"  rate limited; sleeping {wait}s "
                        f"(set --token or GITHUB_TOKEN to raise the limit to 5000/hr)",
                        file=sys.stderr,
                    )
                    time.sleep(min(wait, 900))
                    continue
            if exc.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def truncate_body(body: str | None) -> tuple[str, bool]:
    text = (body or "").replace("\r\n", "\n").strip()
    if len(text) <= BODY_CHAR_BUDGET:
        return text, False
    return text[:BODY_CHAR_BUDGET].rstrip() + "\n[...truncated at ingest...]", True


def normalise(raw: dict[str, Any]) -> dict[str, Any]:
    body, truncated = truncate_body(raw.get("body"))
    return {
        "number": raw["number"],
        "title": (raw.get("title") or "").strip(),
        "body": body,
        "body_truncated": truncated,
        "state": raw["state"],
        "created_at": raw["created_at"],
        "closed_at": raw.get("closed_at"),
        "author_association": raw.get("author_association"),
        "user_login": (raw.get("user") or {}).get("login"),
        "comments": raw.get("comments", 0),
        "state_reason": raw.get("state_reason"),
        "maintainer_labels": sorted(
            lbl["name"] for lbl in raw.get("labels", []) if isinstance(lbl, dict)
        ),
        "html_url": raw["html_url"],
    }


def fetch_issues(token: str | None, max_issues: int) -> list[dict[str, Any]]:
    """Get the open and closed issues, using the Search API.

    Why Search and not /repos/{repo}/issues: the plain endpoint mixes pull requests
    in with the issues and stops after 1000 items. doctl has thousands of PRs, so
    that endpoint can't reach the older issues at all. Search accepts `is:issue`,
    which drops the PRs on their side, and doctl's ~536 issues sit well under
    Search's own 1000-result limit.

    It still asks for open and closed separately, to keep each result set well
    clear of that limit. If the repo ever passes 1000 issues in one state, the fix
    is to ask date range by date range, not to raise per_page.
    """
    collected: dict[int, dict[str, Any]] = {}
    for state in ("open", "closed"):
        page = 1
        while True:
            query = f"repo:{REPO}+is:issue+state:{state}"
            url = (
                f"{API}/search/issues?q={query}"
                f"&per_page={PER_PAGE}&page={page}&sort=created&order=asc"
            )
            print(f"  fetching {state} page {page} ...", file=sys.stderr)
            payload, _ = http_get_json(url, token)
            items = payload.get("items", [])
            total = payload.get("total_count", 0)
            for raw in items:
                if "pull_request" in raw:
                    continue  # belt and braces
                collected[raw["number"]] = normalise(raw)
            if len(items) < PER_PAGE or page * PER_PAGE >= total:
                break
            page += 1
            # Search API is 10 req/min unauthenticated, 30/min authenticated.
            time.sleep(2.0 if token else 7.0)

    ordered = sorted(collected.values(), key=lambda i: i["number"])
    return ordered[:max_issues]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    ap.add_argument("--max-issues", type=int, default=600)
    args = ap.parse_args()

    if not args.token:
        print(
            "note: no GitHub token supplied; unauthenticated limit is 60 req/hr.\n"
            "      Set GITHUB_TOKEN for a smoother run.",
            file=sys.stderr,
        )

    print(f"Ingesting issues from {REPO} ...", file=sys.stderr)
    issues = fetch_issues(args.token, args.max_issues)
    issues.sort(key=lambda i: i["number"])

    # Fingerprint of the text that actually reaches the model. If this changes,
    # every saved result is void, which is exactly what it's for.
    hasher = hashlib.sha256()
    for issue in issues:
        hasher.update(f"{issue['number']}\x1f{issue['title']}\x1f{issue['body']}\x1e".encode())
    corpus_hash = hasher.hexdigest()[:16]

    labeled = sum(1 for i in issues if i["maintainer_labels"])
    snapshot = {
        "schema_version": 1,
        "repo": REPO,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "corpus_hash": corpus_hash,
        "body_char_budget": BODY_CHAR_BUDGET,
        "counts": {
            "issues": len(issues),
            "open": sum(1 for i in issues if i["state"] == "open"),
            "closed": sum(1 for i in issues if i["state"] == "closed"),
            "with_maintainer_labels": labeled,
            "truncated_bodies": sum(1 for i in issues if i["body_truncated"]),
        },
        "issues": issues,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "doctl-issues-snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(snapshot["counts"], indent=2))
    print(f"corpus_hash = {corpus_hash}")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
