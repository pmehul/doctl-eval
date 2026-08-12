"""Talking to the models. Two clients, same interface.

`ServerlessInferenceClient` is the real one. DigitalOcean's API is
OpenAI-compatible, so this is an ordinary chat/completions POST. I use httpx
straight rather than the OpenAI SDK because I only need the one call, and doing it
by hand keeps the retry rules, the error handling and the timeouts visible in this
file instead of hidden in someone else's defaults.

`MockClient` fakes the replies offline. It's here so the harness, the UI, the
concurrency limit and the cost maths can all be exercised without a key or spent
credits. It proves nothing about the models, and the app says so wherever its
numbers show up.

The exercise asks for errors broken down by type. That split earns its keep
because each one needs a different fix:

  rate_limit    429 / 529. Drop concurrency, back off longer, ask for more quota.
  timeout       Nothing came back inside REQUEST_TIMEOUT_S. Raise it, or drop the
                model. On reasoning models it usually means the thinking ran past
                the deadline.
  server_error  A 5xx from their side. Retry, then fail over if it keeps up.
  auth          401 / 403. Fix the key. Never retried, since a bad key doesn't
                get better and retrying just burns the run.
  bad_request   400 / 422. Fix the request. Also never retried.
  parse_error   A 200 with nothing usable in it. Change the prompt or the model.
                Kept apart from a wrong label, because following a format and
                picking the right bucket are different skills.
  network       Connection reset, DNS, TLS.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from . import catalog
from .config import Settings
from .corpus import LABELS, Issue
from .prompting import ParseError, build_messages, parse_label

RETRYABLE = {"rate_limit", "timeout", "server_error", "network"}


@dataclass
class CallResult:
    issue_number: int
    model_id: str
    predicted_label: str | None = None
    confidence: float | None = None
    parse_strategy: str | None = None
    raw_output: str = ""
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usd_per_m_input: float | None = None
    usd_per_m_output: float | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None
    error_type: str | None = None
    error_detail: str | None = None
    attempts: int = 1
    http_status: int | None = None
    simulated: bool = False

    def apply_cost(self) -> None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return
        breakdown = catalog.price_call(self.model_id, self.prompt_tokens, self.completion_tokens)
        self.usd_per_m_input = breakdown["usd_per_m_input"]
        self.usd_per_m_output = breakdown["usd_per_m_output"]
        self.input_cost_usd = breakdown["input_cost_usd"]
        self.output_cost_usd = breakdown["output_cost_usd"]
        self.total_cost_usd = breakdown["total_cost_usd"]


class Client(Protocol):
    async def classify(self, issue: Issue, model_id: str) -> CallResult: ...
    async def aclose(self) -> None: ...


def _classify_http_error(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status in (429, 529):
        return "rate_limit"
    if status in (400, 404, 422):
        return "bad_request"
    if status >= 500:
        return "server_error"
    return "other"


# --------------------------------------------------------------------------
# Real provider
# --------------------------------------------------------------------------


class ServerlessInferenceClient:
    def __init__(self, settings: Settings, concurrency: int) -> None:
        self.settings = settings
        self.concurrency = concurrency
        # One shared client, with the connection pool sized from the concurrency
        # this run is actually using and not the env default. This matters: if the
        # pool is smaller than the number of workers, httpx makes them queue for a
        # connection and the p95 I measure is my own waiting, not the provider's.
        # Reading it from settings would cap a run at concurrency 64 to the pool
        # implied by CONCURRENCY=8 and quietly ruin every latency number.
        limits = httpx.Limits(
            max_connections=max(concurrency * 2, 16),
            max_keepalive_connections=max(concurrency, 8),
        )
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
            limits=limits,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def classify(self, issue: Issue, model_id: str) -> CallResult:
        spec = catalog.get(model_id)
        result = CallResult(issue_number=issue.number, model_id=model_id)
        payload = {
            "model": model_id,
            "messages": build_messages(issue),
            "temperature": self.settings.temperature,
            "max_tokens": (
                self.settings.reasoning_max_tokens if spec.reasoning else self.settings.max_tokens
            ),
        }

        for attempt in range(1, self.settings.max_retries + 1):
            result.attempts = attempt
            started = time.perf_counter()
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                # Latency is recorded per attempt and reflects the attempt the
                # caller actually got an answer from, wall-to-wall.
                result.latency_ms = (time.perf_counter() - started) * 1000
                result.http_status = resp.status_code

                if resp.status_code != 200:
                    kind = _classify_http_error(resp.status_code)
                    result.error_type = kind
                    result.error_detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    if kind in RETRYABLE and attempt < self.settings.max_retries:
                        await self._backoff(attempt, resp)
                        continue
                    return result

                body = resp.json()
                usage = body.get("usage") or {}
                # Prefer provider-reported usage: it is what you are billed on.
                # Estimating tokens locally would produce a cost figure that does
                # not reconcile with the invoice, which is the one thing a cost
                # analysis must not do.
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")

                choices = body.get("choices") or []
                content = ""
                if choices:
                    message = choices[0].get("message") or {}
                    content = message.get("content") or ""
                    # Some providers surface reasoning in a sibling field rather
                    # than inline; keep it for display but never parse from it.
                    reasoning = message.get("reasoning_content")
                    if reasoning and not content:
                        content = reasoning
                result.raw_output = content

                if result.prompt_tokens is None or result.completion_tokens is None:
                    result.error_type = "other"
                    result.error_detail = "provider omitted usage; cost cannot be attributed"
                    return result
                result.apply_cost()

                try:
                    label, conf, strategy = parse_label(content)
                except ParseError as exc:
                    # A 200 with an unusable body. Not retried: at temperature 0
                    # the same request yields the same unusable body, so a retry
                    # spends money to learn nothing.
                    result.error_type = "parse_error"
                    result.error_detail = str(exc)[:300]
                    return result

                result.predicted_label = label
                result.confidence = conf
                result.parse_strategy = strategy
                result.error_type = None
                result.error_detail = None
                return result

            except (httpx.TimeoutException,) as exc:
                result.latency_ms = (time.perf_counter() - started) * 1000
                result.error_type = "timeout"
                result.error_detail = f"{type(exc).__name__}: {exc}"[:300]
            except (httpx.TransportError, httpx.HTTPError) as exc:
                result.latency_ms = (time.perf_counter() - started) * 1000
                result.error_type = "network"
                result.error_detail = f"{type(exc).__name__}: {exc}"[:300]

            if attempt < self.settings.max_retries:
                await self._backoff(attempt, None)

        return result

    async def _backoff(self, attempt: int, resp: httpx.Response | None) -> None:
        """Back off, doubling each time, with the delay picked at random. Obeys
        Retry-After when the server sends one.

        The randomness is the important bit. Every worker hits the same rate limit
        at the same instant, so a fixed delay sends them all back together and
        recreates the burst that caused it.
        """
        if resp is not None:
            hinted = resp.headers.get("retry-after")
            if hinted:
                try:
                    await asyncio.sleep(min(float(hinted), 30.0))
                    return
                except ValueError:
                    pass
        ceiling = self.settings.retry_base_delay_s * (2 ** (attempt - 1))
        await asyncio.sleep(random.uniform(0, min(ceiling, 20.0)))


# --------------------------------------------------------------------------
# Offline simulator
# --------------------------------------------------------------------------

# Behaviour profiles for the simulator, taken from the measured screening run on the
# dev split. Skill is that model's macro-F1 and lat_ms is its measured p50.
#
# These started as my guesses at the ranking, written before any model had been
# called, and they were badly wrong in both directions: mistral-3-14B was guessed
# last of eleven at 0.68 and measured first at 0.816, qwen3.5-397b-a17b was guessed
# first at 0.87 and cannot complete a run at all, and alibaba-qwen3-32b was guessed
# at 780ms against a measured p50 of 15,844ms. Left alone, `make serve-mock` showed
# a leaderboard that contradicted the recommendation in the README, which is a
# confusing thing to hand a reviewer.
#
# Replacing guesses with measurements does not turn simulated output into evidence,
# and it must not be read that way round. These numbers are derived from the results,
# so a mock run agreeing with the recommendation is circular by construction and
# proves nothing. What it buys is a simulator whose shape resembles the real thing,
# so the UI can be developed offline without teaching anyone a false ranking.
_MOCK_PROFILES: dict[str, dict[str, float]] = {
    "openai-gpt-oss-20b": {"skill": 0.781, "lat_ms": 2304, "lat_spread": 0.45},
    "openai-gpt-oss-120b": {"skill": 0.814, "lat_ms": 3101, "lat_spread": 0.40},
    "mistral-3-14B": {"skill": 0.816, "lat_ms": 1335, "lat_spread": 0.40},
    "gemma-4-31B-it": {"skill": 0.809, "lat_ms": 1792, "lat_spread": 0.42},
    "alibaba-qwen3-32b": {"skill": 0.835, "lat_ms": 15844, "lat_spread": 0.42},
    "nvidia-nemotron-3-super-120b": {"skill": 0.786, "lat_ms": 8220, "lat_spread": 0.45},
    "llama3.3-70b-instruct": {"skill": 0.804, "lat_ms": 9063, "lat_spread": 0.48},
    "llama-4-maverick": {"skill": 0.777, "lat_ms": 3340, "lat_spread": 0.46},
    "deepseek-4-flash": {"skill": 0.775, "lat_ms": 2555, "lat_spread": 0.44},
    # Never measured. It rate-limited 105 of 109 calls at concurrency 8 and returned
    # one usable answer, so there is no macro-F1 for it. The field median stands in
    # here purely to keep the simulator running; it is not a claim about the model.
    "qwen3.5-397b-a17b": {"skill": 0.800, "lat_ms": 1031, "lat_spread": 0.50},
    "deepseek-r1-distill-llama-70b": {"skill": 0.812, "lat_ms": 45779, "lat_spread": 0.55},
}

# When a model is wrong it isn't wrong at random. On this task the real mix-ups
# are enhancement against bug, and documentation being called enhancement. Baking
# that in stops the fake confusion matrices being a clean diagonal plus noise.
_CONFUSION_BIAS: dict[str, list[str]] = {
    "bug": ["enhancement", "question", "other"],
    "enhancement": ["bug", "documentation", "question"],
    "question": ["bug", "enhancement", "documentation"],
    "documentation": ["enhancement", "bug", "question"],
    "security": ["bug", "other"],
    "other": ["question", "bug", "enhancement"],
}


class MockClient:
    """Deterministic simulator. Same (issue, model) always yields the same result."""

    def __init__(self, settings: Settings, concurrency: int) -> None:
        self.settings = settings
        self.concurrency = concurrency

    async def aclose(self) -> None:
        return None

    async def classify(self, issue: Issue, model_id: str) -> CallResult:
        spec = catalog.get(model_id)
        profile = _MOCK_PROFILES.get(model_id, {"skill": 0.75, "lat_ms": 800, "lat_spread": 0.4})
        rng = random.Random(
            int(hashlib.sha256(f"{model_id}:{issue.number}".encode()).hexdigest()[:12], 16)
        )
        result = CallResult(issue_number=issue.number, model_id=model_id, simulated=True)

        # Token counts come off the real prompt, so the cost maths below works on
        # realistic sizes. About 3.6 characters per token is a decent rule of
        # thumb for English plus code. It's an estimate and it's labelled as one.
        prompt_chars = sum(len(m["content"]) for m in build_messages(issue))
        result.prompt_tokens = max(64, int(prompt_chars / 3.6))
        result.completion_tokens = (
            rng.randint(320, 1100) if spec.reasoning else rng.randint(14, 30)
        )
        result.apply_cost()

        # Latency with a long tail on the slow side, which is how real inference
        # behaves. It scales with concurrency, so turning the worker count up
        # visibly costs p95, the same way a queue on a real endpoint would.
        base = float(profile["lat_ms"]) * (1 + 0.02 * max(0, self.concurrency - 4))
        result.latency_ms = max(60.0, base * math_lognorm(rng, float(profile["lat_spread"])))

        # Actually wait. Without this nothing ever queues, the wall clock drops to
        # milliseconds and throughput reads in the tens of thousands per second,
        # which is obvious nonsense on screen. Waiting makes the concurrency limit
        # do real work, so throughput lands around concurrency divided by mean
        # latency, which is how the real endpoint behaves. MOCK_TIME_SCALE shrinks
        # the wait so a full 1072-call run takes ~10s instead of ~2min, and it's
        # reported in the results so nobody reads a shrunk clock as a measured one.
        await asyncio.sleep(result.latency_ms / 1000.0 / self.settings.mock_time_scale)

        # Simulate the two error classes that actually bite in production.
        roll = rng.random()
        if roll < 0.004 * max(1, self.concurrency / 8):
            result.error_type = "rate_limit"
            result.error_detail = "simulated HTTP 429"
            result.attempts = 2
            return result
        if roll < 0.006:
            result.error_type = "timeout"
            result.error_detail = "simulated request timeout"
            return result

        truth = issue.gold_label
        if truth is None:
            # No answer key here. Pick one stand-in answer per issue, shared by
            # both models, then let each one hit or miss it on its own skill. If I
            # drew separately for each model they'd be unrelated and agreement
            # would fall to about 0.3, the odds of two coin flips matching. Real
            # models on the same issues agree 70-85% of the time. Agreement is the
            # headline on the unscored tab, so getting this right is the difference
            # between a demo that looks believable and one that's visibly broken.
            truth_rng = random.Random(
                int(hashlib.sha256(f"pseudo-truth:{issue.number}".encode()).hexdigest()[:12], 16)
            )
            truth = truth_rng.choices(
                list(LABELS), weights=[0.44, 0.30, 0.10, 0.07, 0.05, 0.04], k=1
            )[0]

        if rng.random() < float(profile["skill"]):
            label = truth
        else:
            label = rng.choice(_CONFUSION_BIAS.get(truth, list(LABELS)))

        result.predicted_label = label
        result.confidence = round(rng.uniform(0.55, 0.98), 2)
        result.parse_strategy = "bare_json"
        result.raw_output = (
            f'{{"label": "{label}", "confidence": {result.confidence}}}'
            if not spec.reasoning
            else f"<think>Weighing the title against the body...</think>\n"
                 f'{{"label": "{label}", "confidence": {result.confidence}}}'
        )

        # Break the format on purpose now and then. Models ignoring the output
        # format is a real production risk, so the parse_error path should be
        # visible on screen and not only in a test.
        if rng.random() < 0.01:
            result.raw_output = f"I think this is a {label} report."
            result.predicted_label = label
            result.parse_strategy = "loose_scan"
            result.confidence = None

        return result


def math_lognorm(rng: random.Random, sigma: float) -> float:
    """Multiplier with median 1.0 and a right tail, for latency simulation."""
    return float(pow(2.718281828459045, rng.gauss(0.0, sigma)))


def make_client(settings: Settings, concurrency: int) -> Client:
    """Build a client for a run at a given concurrency.

    Concurrency is passed in instead of read from settings, because the UI can
    change it per run. Anything that scales with it, the connection pool and the
    simulator's queueing, has to see the value the run actually used, or the p95
    you get back belongs to a different experiment.
    """
    if settings.is_mock:
        return MockClient(settings, concurrency)
    return ServerlessInferenceClient(settings, concurrency)
