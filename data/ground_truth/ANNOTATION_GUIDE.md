# How the labels were decided

This is the rulebook the answer key was written against. It exists because "bug or
enhancement?" isn't obvious on a real repo, and a test whose answers have no written
rules is partly measuring the mood of whoever labelled it.

The same six categories and the same tie-break rules go into the models' system
prompt, so the models and the answer key are held to one definition. Marking a model
down for conventions nobody told it about tests mind-reading, not skill.

## The six categories

**bug** — What happened isn't what the docs or the obvious intent say should happen.
Crashes, wrong output, flags that get ignored, auth that fails, released builds that
are broken or wrongly named.

**enhancement** — Asking for something that doesn't exist, or an improvement to
something already working as designed. New commands and flags, new places to download
it from (a FreeBSD build, a Debian repo), extra output formats, and internal work like
linting, formatting, refactors and CI.

**question** — The author wants to know how to use it or how it behaves. No defect
claimed, no change asked for. The right ending is an answer, not a commit.

**documentation** — The fix is words someone reads, not behaviour. README, docs site,
help output, error and warning text that's plain wrong. Missing docs count.

**security** — Reports a vulnerability or something security-relevant. CVE reports
against dependencies, exposed credentials, unsafe defaults.

**other** — Fits none of the above. Spam, empty or placeholder issues, duplicates,
off-topic posts, admin requests, and anything too ambiguous to place.

## Tie-break rules

These are the cases where two people actually disagree, so they're decided up front
instead of issue by issue.

1. **Words or behaviour.** If the fix changes text a user reads (help output, README,
   an error message that's simply wrong), it's `documentation`. If it changes what the
   program does, it's `bug` or `enhancement`.

2. **Broken build or missing build.** A published build that won't run, has the wrong
   checksum, or is named wrongly is a `bug`. Asking for a build that was never
   published (MacPorts, FreeBSD, ARM) is an `enhancement`.

3. **A question that's really a request.** "Does doctl support X?" where X doesn't
   exist is a request in disguise. If the author only wants an answer, it's a
   `question`. If they want X built, it's an `enhancement`. Where the text honestly
   supports both readings, the issue is left out rather than forced.

4. **Internal work** (gofmt, golangci-lint, regenerating mocks, dependency bumps) is
   an `enhancement`, not `other`. Someone is asking for work on the codebase.

5. **The body beats the title.** doctl #26 is titled "What broke Jim?" and is a
   straightforward authentication bug.

6. **Caring about security isn't the security category.** "Don't write my token to
   disk" is a hardening `enhancement`. `security` is for reports of an actual
   weakness. Anything sitting on that line is left out.

7. **Ambiguity gets recorded, not guessed at.** An issue two sensible people would
   split on is left out, with the reason written in `hand_labels.json`. Forcing those
   in adds noise that randomly punishes whichever model read it the other way.

## What this answer key can't support

Said plainly, because it bounds what the numbers mean.

- **One person labelled it, once.** So there's no figure for how often two people
  would agree, which means the ceiling on this task is unmeasured. About 15 issues out
  of roughly 100 examined were left out as ambiguous, which suggests two people would
  disagree somewhere in the low teens percent, and therefore that the real ceiling is
  well under 100%. Any model in the low 90s is at the noise floor of this data, and a
  gap of one or two points between models means nothing.

- **The security category is almost all bot output.** All 26 issues with the
  `security vulnerability` label are templated Mend/WhiteSource CVE reports shaped
  like "CVE-XXXX-YYYY (Severity) detected in <package>". A regex can spot them, so a
  `security` score measures pattern matching rather than security judgement. They're
  flagged `templated: true` and macro-F1 is reported both with and without them.

- **The maintainers' labels aren't a gold standard.** Different people applied them
  over about ten years. `suggestion` and `enhancement` both exist meaning the same
  thing. `docs` was used five times in the repo's whole history. They're treated as
  good but imperfect, and where my own labelling disagrees with them, the disagreement
  is recorded rather than hidden.

- **The categories are very uneven.** `bug` and `enhancement` dominate. Accuracy on
  its own would reward a model that ignores the four small ones, so macro-F1 is the
  headline and the number of examples is shown next to every score.
