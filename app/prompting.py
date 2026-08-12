"""Builds the prompt, and reads the answer back out.

Every model gets the same prompt. If I tuned the prompt per model I'd be
comparing prompts, not models, and the customer's question is which model to put
behind the prompt they already have.

The prompt spells out the same tie-break rules I used when labelling by hand
(they live in ANNOTATION_GUIDE.md). Marking a model down for conventions nobody
told it about tests mind-reading, not skill.

Four worked examples go in, all from the dev split. They cover the boundaries
that actually trip things up: help text vs behaviour, a question that's really a
feature request, and off-topic vs bug. Keeping them out of the test split matters
because an example the model has already seen is memorised, not classified.

I ask for a small JSON object back. Not every model here supports a strict
output mode, and demanding one would knock candidates out for the wrong reason,
so instead the parser tries a few things in turn: JSON in a code fence, bare
JSON, JSON buried in a sentence, then a plain scan for a label. If none work
that's a `parse_error`, counted apart from a wrong label, because "can't hold a
format" and "picks the wrong bucket" are different problems with different fixes.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from .corpus import LABELS, Issue

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """\
You are an issue triage classifier for a software repository. You assign exactly \
one category to each GitHub issue.

Categories:
- bug: reported behaviour deviates from documented or intended behaviour. Crashes, \
panics, wrong output, ignored flags, failing auth, broken or mis-named release artefacts.
- enhancement: a request for a capability that does not exist, or an improvement to \
one that already works as designed. New commands or flags, new distribution channels, \
output additions, and internal engineering work such as linting, formatting, refactors and CI.
- question: the author wants to know how to use the tool or how it behaves. No defect \
is claimed and no change is requested. The right resolution is an answer, not a commit.
- documentation: the fix lands in text rather than behaviour. README, docs, help output, \
or an error/warning string that is factually wrong. Missing documentation counts.
- security: reports a vulnerability or security-relevant weakness, including CVE reports \
against dependencies, credential exposure, or insecure defaults.
- other: genuinely fits none of the above. Spam, placeholder or empty issues, duplicates, \
off-topic posts, administrative requests.

Tie-break rules, in order:
1. If the fix is to change words a user reads, choose documentation. If the fix is to \
change what the program does, choose bug or enhancement.
2. A published build that does not run, has a wrong checksum, or is mis-named is a bug. \
A request for a build that was never published is an enhancement.
3. "Does it support X?" where the author only wants an answer is a question. Where they \
want X built, it is an enhancement.
4. Internal engineering work on the codebase is an enhancement, not other.
5. The body outweighs the title. Titles are often misleading.
6. Security motivation alone is not the security class. "Do not store my token on disk" \
is an enhancement; security is for reports of an actual weakness.

Respond with only a JSON object, no prose and no code fence:
{"label": "<one of: bug, enhancement, question, documentation, security, other>", "confidence": <0.0-1.0>}\
"""

# Few-shot examples, referenced by issue number and loaded from the corpus rather
# than pasted in as literals. Two reasons: the examples stay honest (they are real
# issues with their real gold labels, not idealised paraphrases), and the loader
# asserts every one is in the *dev* split, so it is structurally impossible to
# leak a test item into the prompt. An earlier draft of this file hardcoded four
# paraphrased examples and three of them turned out to be test-split issues --
# hence the assertion rather than a comment asking future me to be careful.
#
#   446  titled "Suggestion: ..." but the fix is help text  -> teaches rule 1
#   293  "is there a way to see logs...?"                    -> teaches rule 3
#   751  asks for a Debian repo that was never published     -> teaches rule 2
#  1363  spam, no content                                    -> teaches `other`
FEW_SHOT_NUMBERS: tuple[int, ...] = (446, 293, 751, 1363)

# Few-shot bodies are trimmed harder than the item under test. They are fixed
# overhead paid on every single call, so 500 chars each keeps the standing input
# cost predictable instead of letting one verbose example tax the whole run.
FEW_SHOT_BODY_CHARS = 500


def _issue_block(title: str, body: str) -> str:
    body = (body or "").strip() or "(no body provided)"
    return f"TITLE: {title}\n\nBODY:\n{body}"


@lru_cache(maxsize=1)
def few_shot_messages() -> tuple[dict[str, str], ...]:
    from .corpus import load_corpus  # local import: avoids an import cycle

    corpus = load_corpus()
    out: list[dict[str, str]] = []
    for number in FEW_SHOT_NUMBERS:
        issue = corpus.by_number(number)
        if issue is None:
            raise RuntimeError(f"few-shot issue #{number} is not in the corpus snapshot")
        if issue.gold_label is None:
            raise RuntimeError(f"few-shot issue #{number} has no gold label")
        if issue.gold_split != "dev":
            raise RuntimeError(
                f"few-shot issue #{number} is in the {issue.gold_split!r} split. "
                "Few-shot examples must come from dev only, otherwise reported "
                "test accuracy is partly memorisation."
            )
        body = issue.body.strip()[:FEW_SHOT_BODY_CHARS]
        out.append({"role": "user", "content": _issue_block(issue.title, body)})
        out.append(
            {
                "role": "assistant",
                "content": json.dumps({"label": issue.gold_label, "confidence": 0.9}),
            }
        )
    return tuple(out)


def build_messages(issue: Issue) -> list[dict[str, str]]:
    """One request per issue. Never batched.

    Batching would be cheaper per issue, and the exercise rules it out for
    reasons I'd have landed on anyway. You lose per-issue timing. You can't retry
    one failure without redoing the whole batch. One bad reply can spoil the ones
    next to it. And you can't send a single hard issue to a stronger model, which
    is the thing the production plan is built on.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(dict(m) for m in few_shot_messages())
    messages.append({"role": "user", "content": _issue_block(issue.title, issue.body)})
    return messages


# --- parsing -------------------------------------------------------------

# Reasoning models put their thinking in <think>...</think> and the answer after
# it. That block has to come off before anything else: the thinking almost always
# mentions several labels on the way, so a plain scan picks up the wrong one.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class ParseError(ValueError):
    """Raised when no label can be recovered from a completion."""


def strip_reasoning(text: str) -> str:
    text = _THINK_RE.sub("", text)
    # Truncated reasoning: an opening tag with no close (hit the token ceiling).
    if "</think>" in text:
        text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def parse_label(raw: str) -> tuple[str, float | None, str]:
    """Pull (label, confidence, parse_strategy) out of the model's reply.

    It returns which route worked so the UI can show *how* each answer was read.
    A model that only ever gets through on the last-resort scan isn't really
    following instructions, and you want to know that before you ship it.
    """
    if not raw or not raw.strip():
        raise ParseError("empty completion")

    text = strip_reasoning(raw)
    if not text:
        raise ParseError("completion contained only reasoning tokens (raise REASONING_MAX_TOKENS)")

    candidates: list[tuple[str, str]] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(("fenced_json", m.group(1).strip()))
    candidates.append(("bare_json", text))
    for m in _OBJ_RE.finditer(text):
        candidates.append(("embedded_json", m.group(0)))

    for strategy, blob in candidates:
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        label = obj.get("label")
        if isinstance(label, str) and label.strip().lower() in LABELS:
            conf = obj.get("confidence")
            conf_f = float(conf) if isinstance(conf, (int, float)) else None
            if conf_f is not None:
                conf_f = min(1.0, max(0.0, conf_f))
            return label.strip().lower(), conf_f, strategy

    # Last resort: the model wrote a sentence instead. Take the first label that
    # shows up as a whole word. Logged as `loose_scan` so it's visible in the UI.
    lowered = text.lower()
    hits = [(lowered.find(lbl), lbl) for lbl in LABELS if re.search(rf"\b{lbl}\b", lowered)]
    hits = [(pos, lbl) for pos, lbl in hits if pos >= 0]
    if hits:
        hits.sort()
        return hits[0][1], None, "loose_scan"

    raise ParseError(f"no schema label found in completion: {text[:200]!r}")
