"""The models being compared, and what they cost.

Two things share this one small file so both are easy to check.

First, the list of models, each with the facts that put it there: size, whether the
whole model runs per word or only a slice, and whether it thinks out loud first. The
list only has DigitalOcean-hosted models on it, because the exercise credits don't
work anywhere else.

Second, the prices, written as dollars per million tokens exactly as published. All
the arithmetic is in one function, `price_call`, so you can check a dollar figure by
hand instead of trusting a total. Every saved result carries the prices it was
worked out with, so a later price change can't rewrite an old conclusion.

Prices copied from the DigitalOcean-hosted and OpenAI open-weight sections of the
Inference pricing page, checked 7 Aug 2026:
https://docs.digitalocean.com/products/inference/details/pricing/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

PRICING_SOURCE = "https://docs.digitalocean.com/products/inference/details/pricing/"
PRICING_VERIFIED = "2026-08-07"


@dataclass(frozen=True)
class ModelSpec:
    id: str                     # slug sent to the API as `model`
    label: str                  # human name for the UI
    vendor: str
    params: str                 # as published; MoE models note active params
    architecture: str           # "dense" | "moe"
    usd_per_m_input: float
    usd_per_m_output: float
    reasoning: bool             # emits thinking tokens before the answer
    context_window: int
    why_included: str           # what this candidate is in the pool to test


# The models being compared. The list is meant to cover a wide range rather than
# to be complete: sizes from 14B to 397B, models where the whole thing runs on
# every word against models where only a slice runs, one model that thinks out
# loud before answering, and a roughly 15x spread in price.
#
# `why_included` says, in plain words, what question each model is here to answer.
# It shows up in the UI, so it is written for someone reading the app rather than
# reading this file.
CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="openai-gpt-oss-20b",
        label="gpt-oss-20b",
        vendor="OpenAI (open weight)",
        params="~21B total / ~3.6B active",
        architecture="moe",
        usd_per_m_input=0.05,
        usd_per_m_output=0.45,
        reasoning=False,
        context_window=128_000,
        why_included="The cheapest one on the list. If even this can do the job, there is "
                     "no reason to pay for anything bigger.",
    ),
    ModelSpec(
        id="openai-gpt-oss-120b",
        label="gpt-oss-120b",
        vendor="OpenAI (open weight)",
        params="~117B total / ~5.1B active",
        architecture="moe",
        usd_per_m_input=0.10,
        usd_per_m_output=0.70,
        reasoning=False,
        context_window=128_000,
        why_included="Should give big-model quality at a small-model price, because only a "
                     "small slice of it runs for each word. My favourite going in.",
    ),
    ModelSpec(
        id="mistral-3-14B",
        label="Ministral 3 14B",
        vendor="Mistral AI",
        params="14B",
        architecture="dense",
        usd_per_m_input=0.20,
        usd_per_m_output=0.20,
        reasoning=False,
        context_window=262_144,
        why_included="A small model where the whole thing runs every time. Tests whether "
                     "that beats a bigger model that only runs a slice, at about the same "
                     "price.",
    ),
    ModelSpec(
        id="gemma-4-31B-it",
        label="Gemma 4 31B",
        vendor="Google",
        params="31B",
        architecture="dense",
        usd_per_m_input=0.18,
        usd_per_m_output=0.50,
        reasoning=False,
        context_window=256_000,
        why_included="Built by a different company on different data, so it should get "
                     "different questions wrong. Useful when I want two models that fail in "
                     "different places.",
    ),
    ModelSpec(
        id="alibaba-qwen3-32b",
        label="Qwen3-32B",
        vendor="Alibaba",
        params="32.8B",
        architecture="dense",
        usd_per_m_input=0.25,
        usd_per_m_output=0.55,
        reasoning=False,
        context_window=32_768,
        why_included="The mid-size model people normally reach for. Here as the sensible "
                     "default that anything else has to beat.",
    ),
    ModelSpec(
        id="nvidia-nemotron-3-super-120b",
        label="Nemotron 3 Super 120B",
        vendor="NVIDIA",
        params="120B",
        architecture="dense",
        usd_per_m_input=0.165,
        usd_per_m_output=0.358,
        reasoning=False,
        context_window=1_000_000,
        why_included="A 120B model that costs less to run than the 32B ones. If it is any "
                     "good, it beats them on price outright.",
    ),
    ModelSpec(
        id="llama3.3-70b-instruct",
        label="Llama 3.3 70B Instruct",
        vendor="Meta",
        params="70B",
        architecture="dense",
        usd_per_m_input=0.65,
        usd_per_m_output=0.65,
        reasoning=False,
        context_window=128_000,
        why_included="The one customers ask for by name. Here so I have a real answer when "
                     "they do, instead of an opinion.",
    ),
    ModelSpec(
        id="llama-4-maverick",
        label="Llama 4 Maverick",
        vendor="Meta",
        params="400B total / 17B active",
        architecture="moe",
        usd_per_m_input=0.20,
        usd_per_m_output=0.696,
        reasoning=False,
        context_window=128_000,
        why_included="Much bigger overall than the others. Tests whether sheer size helps "
                     "at all when the text being sorted is this short.",
    ),
    ModelSpec(
        id="deepseek-4-flash",
        label="DeepSeek V4 Flash",
        vendor="DeepSeek",
        params="284B total",
        architecture="moe",
        usd_per_m_input=0.068,
        usd_per_m_output=0.168,
        reasoning=False,
        context_window=1_048_576,
        why_included="Cheapest to run of the whole list, and still a big model. The one "
                     "most likely to quietly win.",
    ),
    ModelSpec(
        id="qwen3.5-397b-a17b",
        label="Qwen 3.5 397B A17B",
        vendor="Alibaba",
        params="397B total / 17B active",
        architecture="moe",
        usd_per_m_input=0.302,
        usd_per_m_output=1.925,
        reasoning=False,
        context_window=131_072,
        why_included="The best of the models I am allowed to use here. It stands in for the "
                     "expensive model the customer runs today, so I can see what they would "
                     "give up by switching.",
    ),
    ModelSpec(
        id="deepseek-r1-distill-llama-70b",
        label="DeepSeek R1 Distill Llama 70B",
        vendor="DeepSeek",
        params="70B",
        architecture="dense",
        usd_per_m_input=0.99,
        usd_per_m_output=0.99,
        reasoning=True,
        context_window=32_678,
        why_included="This one writes out its thinking before answering, about 30 times "
                     "more text than the others. That costs roughly 45 times more, so it is "
                     "here to find out whether the thinking is worth paying for.",
    ),
)

BY_ID: dict[str, ModelSpec] = {m.id: m for m in CATALOG}

# Defaults for the side-by-side view. These are the two models the README
# recommends, so the application opens on the recommendation rather than on an
# arbitrary pair: a cheap sparse workhorse as primary against a near-frontier open
# weight as the escalation tier. Different vendors and different pretraining
# lineages, so their errors should decorrelate -- which is what makes
# model-vs-model disagreement usable as a routing signal. ~3x apart on input price
# and ~2.75x on output: a real tradeoff, not two sizes of one model.
DEFAULT_MODEL_A = "openai-gpt-oss-120b"
DEFAULT_MODEL_B = "qwen3.5-397b-a17b"


def get(model_id: str) -> ModelSpec:
    try:
        return BY_ID[model_id]
    except KeyError:
        raise KeyError(
            f"unknown model {model_id!r}; available: {', '.join(sorted(BY_ID))}"
        ) from None


def price_call(model_id: str, prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    """What one call cost, with every step shown.

    It hands back the whole breakdown, not a single number. You should be able to
    read `prompt_tokens`, `usd_per_m_input` and `input_cost_usd` off one object and
    check the multiplication yourself, for any call, without reading this function.

        input_cost  = prompt_tokens     / 1_000_000 * usd_per_m_input
        output_cost = completion_tokens / 1_000_000 * usd_per_m_output
        total       = input_cost + output_cost

    Nothing is rounded. One call costs about $0.00001, so rounding to cents here
    would make every call $0.00 and the totals would come out wrong. Rounding only
    happens on screen.
    """
    spec = get(model_id)
    input_cost = prompt_tokens / 1_000_000 * spec.usd_per_m_input
    output_cost = completion_tokens / 1_000_000 * spec.usd_per_m_output
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usd_per_m_input": spec.usd_per_m_input,
        "usd_per_m_output": spec.usd_per_m_output,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def catalog_payload() -> dict[str, object]:
    return {
        "pricing_source": PRICING_SOURCE,
        "pricing_verified": PRICING_VERIFIED,
        "default_model_a": DEFAULT_MODEL_A,
        "default_model_b": DEFAULT_MODEL_B,
        "models": [asdict(m) for m in CATALOG],
    }
