# Should this customer pay for a frontier model to sort their issues?

**Running app: https://seal-app-34zea.ondigitalocean.app/**

The app asks for a username and password. The username is `reviewer`, and the password is in
the covering email. That login exists because `POST /api/run` spends real inference credits and
has no rate limit, so an open URL could be drained by anyone who found it.

My answer is no, and this document explains why.

The customer automatically sorts incoming GitHub issues into categories. They do this at high
volume, across many repositories, and every issue is sent to an expensive frontier model. They
suspect they are wasting money, and I think they are correct.

I used the doctl repository as a test bed. It gave me 536 real issues, which I sorted into six
categories, running every issue through two different models so that the two could be compared
fairly. doctl itself is not important here. It is simply a convenient source of real data. The
method does not depend on doctl, on these six categories, or on DigitalOcean, and that
portability matters because the customer will want to apply the same method to their other
repositories next.

**My recommendation is to send every issue to `mistral-3-14B`, and to switch to
`deepseek-4-flash` once volume passes roughly ten million classifications a month.** Both are
hosted by DigitalOcean, so the customer keeps one provider and one API key. I explain the
reasoning below.

That is not the recommendation I expected to make. I began with a plan to run a cheap model on
everything and escalate the hard issues to a large one. The measurements killed that plan, and
the section on trade-offs says exactly how, because I wrote down in advance the result that
would make me abandon it.

One point before anything else: the application in this repository *is* the test. Every number
shown in the application is produced by the code here. I did not calculate results separately
and paste them in.

## Please read this before you read any numbers

Every figure below comes from real calls to DigitalOcean Serverless Inference. Around 4,000 of
them, for about $1.30. Saved results are named `-live-`, and a file named `-sim-` came from the
offline simulator and is not evidence. There are none of those left in this repository.

One thing matters more than any single number here, so I want it stated before the results
rather than buried after them. **A run is not reproducible, even at temperature 0.** I ran the
same model over the same 109 issues six times and got macro-F1 between 0.774 and 0.810. One
configuration produced 0.847 on one occasion and 0.774 on another. A hosted endpoint batches
work across requests, and that changes the arithmetic slightly, so identical inputs do not give
identical outputs.

The consequence is uncomfortable. My first screening run ranked eleven models by macro-F1 to
three decimal places, first place 0.847 down to tenth 0.777, a spread of 0.070. Repeated
measurement then showed a single model varying by 0.036 on its own. **The noise was about as
wide as the entire ranking**, so most of that ordering was meaningless, and the top two were in
the wrong order. The model I now recommend was third in that table.

So every quality claim below rests on five runs per model, and I say plainly where a difference
is too small to call. Where I can only offer one run, I say that too.

## The models I tested

I tested eleven models.

The exercise credits only work on models that DigitalOcean hosts itself, so every model here is
open-weight. That means I have no direct comparison against a frontier model, and I am not
claiming one. `qwen3.5-397b-a17b` was meant to stand in for whatever expensive model the customer
pays for today, and it could not: it returned one usable answer out of 109 and rate-limited the
rest. So I have no proxy for the frontier tier at all, and I say what follows without one.

That is less of a hole than it sounds. The customer's question is whether a cheaper model is good
enough, and the strongest evidence for "yes" is that the models here are indistinguishable from
*each other* across a 20x range of size and an 18x range of price. When a 14B model matches a
284B model on a task, the task is not what is limiting the result, and buying a larger model is
unlikely to change it.

I was not trying to test every available model. I was trying to cover a wide range, so that
whichever model won, I would understand the reason it won. The eleven therefore span three
things. They range from 14 billion parameters up to 397 billion. Some of them run their entire
network for every word they produce, while others run only a small part of it. And their input
prices differ by roughly fifteen times between the cheapest and the most expensive.

The table below needs one word of explanation first. Several of these models are built so that
only part of the network runs for each word, which is how a very large model can still be
inexpensive. Where that is the case, I give both the total size and the part that actually runs,
because the part that runs is what determines how much thinking the model does.

| model | size | how much runs per word | $/M in | $/M out | why I included it |
|---|---|---|---|---|---|
| `openai-gpt-oss-20b` | ~21B total, ~3.6B active | part of it | 0.05 | 0.45 | It is the cheapest model available. If it can do this job, then no larger model is worth paying for. |
| `openai-gpt-oss-120b` | ~117B total, ~5.1B active | part of it | 0.10 | 0.70 | It should give the quality of a 120B model at close to the price of a 20B one. I expected it to win. |
| `mistral-3-14B` | 14B | all of it | 0.20 | 0.20 | It tests whether a small model that runs completely beats a larger one that runs only partly, at a similar price. |
| `gemma-4-31B-it` | 31B | all of it | 0.18 | 0.50 | It was trained by a different company on different data, so it should make different mistakes from the others. |
| `alibaba-qwen3-32b` | 32.8B | all of it | 0.25 | 0.55 | It is the mid-sized model people usually choose by default, so it is the sensible standard that others must beat. |
| `nvidia-nemotron-3-super-120b` | 120B | all of it | 0.165 | 0.358 | It is a 120B model priced below the 32B models. If its quality holds up, it beats them on price outright. |
| `llama3.3-70b-instruct` | 70B | all of it | 0.65 | 0.65 | Customers ask about this model by name, so I wanted a measured answer ready when they do. |
| `llama-4-maverick` | 400B total, 17B active | part of it | 0.20 | 0.696 | It tests whether a much larger total size helps at all when the text being sorted is this short. |
| `deepseek-4-flash` | 284B | part of it | 0.068 | 0.168 | It is the cheapest model here to run and still a very large one, so it could beat my expected winner on cost. |
| `qwen3.5-397b-a17b` | 397B total, 17B active | part of it | 0.302 | 1.925 | It is the strongest model available to me, so it sets the quality ceiling and stands in for the customer's current model. |
| `deepseek-r1-distill-llama-70b` | 70B | all of it, and writes out its reasoning | 0.99 | 0.99 | It tests whether writing out reasoning first is worth its cost on a six-way choice. |

The prices come from
[DigitalOcean's pricing page](https://docs.digitalocean.com/products/inference/details/pricing/)
and I checked them on 7 Aug 2026. They are written down in one file,
[`app/catalog.py`](app/catalog.py), and copied onto every saved result. If DigitalOcean changes
a price later, that change therefore cannot quietly alter a conclusion I reached earlier.

**Models I chose not to include.** I left out `qwen3-coder-flash` because it is built for
reading code and this task is reading English prose. I left out the Kimi, GLM and
Nemotron-Ultra models because they cost between $0.75 and $2.85 per million input tokens, which
is roughly ten times my main recommendation, for work the cheaper models can probably do. I
left out the image, audio and video models because they do not perform this kind of task at all.
And I could not include Claude or GPT-5.x, because the exercise credits do not cover them.

**Why I included the model that writes out its reasoning.** I included
`deepseek-r1-distill-llama-70b` so that I could rule it out with evidence, rather than leaving
it off the list because I assumed it would lose. The claim that reasoning models are
unnecessary for classification is only an opinion until someone puts a price on it.

So here is the price. That model writes roughly 700 tokens of reasoning before it gives an
answer, where the other models write about 22 tokens in total. At its published rate, those
extra tokens cost roughly **45 times as much as the other models spend on output**, and all of
that expense buys one word chosen from a list of six. It also does its reasoning while the
caller waits, so a request that takes a fraction of a second with other models takes several
whole seconds with this one. That difference matters because it changes the design from
answering immediately to placing requests in a queue. If this model turns out to win by a large
margin I will change my recommendation, but only as far as using it on the difficult issues.

## What I recommend

**Run `mistral-3-14B` on everything. Move to `deepseek-4-flash` when volume makes the price
difference matter.**

|  | recommended | cheaper alternative |
|---|---|---|
| model | `mistral-3-14B` | `deepseek-4-flash` |
| size | 14B dense | 284B total, MoE |
| built by | Mistral AI | DeepSeek |
| $ per million tokens, in / out | 0.20 / 0.20 | 0.068 / 0.168 |
| accuracy score (macro-F1), 5 runs on dev | **0.816 ± 0.007** | 0.775 ± 0.010 |
| macro-F1 on the held-out test split | **0.758** | 0.727 |
| accuracy on the held-out test split | **86.2%** | 85.8% |
| speed (p50 / p95 latency at concurrency 16) | **1,253 / 2,207 ms** | 2,156 / 5,805 ms |
| cost per correct answer | $0.000398 | **$0.000138** |
| failed calls | 0.0% | 0.0% |

**The honest summary of that table: Mistral is faster, DeepSeek is cheaper, and on quality I
cannot separate them on the work that matters.**

The quality columns need reading carefully, because on the work that matters these two models
are level. On the 253 test issues they choose the same label 88.9% of the time. Both were right
on 205 and both wrong on 23. They differ on which one was right for 25 issues, Mistral winning 13
and DeepSeek 12. A paired test on that gives **p = 1.00**. There is no accuracy difference here
at all.

The remaining 0.031 of macro-F1 comes from a single class. Macro-F1 averages the six classes
equally, so `documentation`, with 13 issues, counts as much as `bug`, with 121. Mistral scores
0.67 there and DeepSeek 0.46, and that one class more than accounts for the whole gap, because
the other five very slightly favour DeepSeek.

I ran the full test split twice, and the second run is the better argument for reading that gap
as noise. Same two models, same 253 issues, temperature 0:

| | first run | second run |
|---|---|---|
| Mistral macro-F1 | 0.758 | 0.758 |
| DeepSeek macro-F1 | 0.647 | **0.727** |
| gap | 0.111 | **0.031** |
| paired test | p = 0.29 | **p = 1.00** |

DeepSeek moved 0.080 between two runs of identical work, mostly because it got one of the three
`other` issues right the second time instead of none. **The gap between the models is smaller
than the gap between one model and itself.**

So I recommend Mistral on speed, not on accuracy. It is 1.7 times faster at p50 and 2.6 times
faster at p95, and it held that lead across five dev runs and both test runs. That is a real
difference. Three points of macro-F1 resting on 13 issues is not.

**When to switch.** Per correct answer DeepSeek costs $0.00014 and Mistral $0.000397, so
DeepSeek is 2.8 times cheaper per useful output even after its lower accuracy is counted. At a
million classifications a month that is $81 against $239, and the $158 difference does not
justify giving up half your latency. At a hundred million it is $8,140 against $23,900, and the
argument reverses. The crossover is somewhere around ten million a month, and I would move at
that point rather than earlier.

**Why two models from different companies.** An outage or a rate limit at one company should not
stop the workload. That is not a hypothetical: during screening `qwen3.5-397b-a17b` failed 105
of 109 calls to rate limiting at concurrency 8, and returned one usable answer. A recommendation
resting on a single provider has no answer to that. 14B dense against 284B mixture-of-experts,
two companies, two architectures, so their capacity limits are unrelated.

## The trade-offs I am making

Three of these were predictions with a number attached, written before I had results. All three
have now been settled, and two of them went against me. I have left them as they were rather
than quietly editing the prediction to match the outcome.

**I said I would abandon the second opinion if the two models scored within 0.03 of each other.
They did, so I abandoned it.** That was the plan the whole design was built around: cheap model
on everything, hard issues escalated to a large one. It does not survive the measurements. The
top six models sit inside a 0.07 band that is no wider than one model's own run-to-run noise, so
there is no larger model here that is reliably better to escalate *to*. Worse, every one of the
eleven scores between 0.44 and 0.67 on `documentation`, the hardest real class. The models fail
on the same issues, which is exactly the case where a second model adds cost and no information.
Hard issues need a person or a better prompt, not more parameters. I would run one model and
route low-confidence issues to a human.

**I said `deepseek-4-flash` should replace the main model on price if it came within 0.02.**
Against Mistral it is 0.041 behind on the dev split with no overlap across five runs each, so on
my own rule it does not qualify as an equal-quality substitute today. It becomes the right choice
on volume instead, for the cost reason above, and that is a different argument from the one I
expected to be making.

**I said reasoning models were very unlikely to be worth paying for. That one held.**
`deepseek-r1-distill-llama-70b` cost $0.00148 per call, 18 times the cheapest model, for a lower
macro-F1 of 0.812, at a p95 of 115 seconds against a 120-second timeout, which is why three of
its calls failed outright. Sorting six labels does not reward deliberation.

**I am trading speed against volume.** Concurrency controls this: how many requests are in
flight at once. I expected the familiar curve, where throughput flattens, individual requests
slow down and rate-limit errors start. **I did not find it.**

| concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| requests per second | 0.35 | 0.81 | 1.22 | 2.62 | 5.80 |
| p95 latency (ms) | 5,806 | 5,618 | 5,218 | 7,412 | 3,794 |

Throughput rose almost in step with concurrency across a 16x range, p95 did not degrade, and
there were no rate-limit errors at all. So the ceiling I hit was my own setting, not the
provider's capacity, and I have no measured saturation point to report.

I stopped at 16 rather than guessing higher, because above that the measurement stops being
honest. 109 issues at concurrency 128 puts every request in flight at once, the corpus runs out,
and wall-clock collapses to the duration of the slowest single call: I measured 5.4 seconds of
wall-clock against a 5.4 second slowest call. That is a burst, not sustained throughput, and
printing it in the same column as the lower numbers would invite a comparison it cannot support.
The harness now refuses to, and warns instead.

I recommend concurrency 16 on that basis: the highest level this corpus can measure properly,
with at least seven full waves of requests behind every figure. A larger corpus would very
likely justify more.

One number from the sweep is worth keeping for operational reasons. At concurrency 32 a single
retry stretched wall-clock from about 15 seconds to 64 and dropped measured throughput from 15.6
to 1.7 requests per second, while p50, p95 and the slowest call all looked completely normal.
Retry backoff is charged to wall-clock and to nothing else, so one retry in 109 calls moved the
headline throughput figure by a factor of nine. The application reports retries next to
throughput for that reason.

This is also why the application always shows p50 and p95 beside the concurrency they were
measured at. A latency figure without the load that produced it cannot be acted on.

**I am trading the value of reasoning against the cost of reasoning.** I covered the figures
above: roughly 45 times the output cost, and a p95 latency measured in whole seconds, to choose
one word from a list of six. It is very unlikely to be worth paying for, and that is a finding
rather than an assumption, because the model is in the test and will be measured.

**Finally, I made trade-offs in the test itself.** Only one person wrote the answer key, and
that person was me. I never checked whether the models' confidence scores actually mean
anything. There is no frontier model to compare against. Each of these is a genuine limitation,
each was a conscious decision about where to spend limited time, and I list all three at the end
of this document.

## What the test showed

I screened eleven models on 109 development issues, then ran the two finalists over all 536
issues, scoring against the 253 held-back test issues. The full table is in
`data/screening/`, the final run is in `data/runs/`, and `make screen` regenerates it. These
were the four things I set out to learn.

**Where the line between price and quality bends. It bends immediately.** The top six models sit
between 0.777 and 0.847 macro-F1, and a single model repeated six times varies by 0.036 on its
own. The band is no wider than the noise, so I cannot rank the models inside it, and the answer
to the customer's question is not "buy the best one". Nothing here justifies paying more for
quality, so the choice comes down to speed and price, which I *can* measure apart. The cheapest
model in the field, `deepseek-4-flash` at $0.068 per million input tokens, is among the best on
quality.

**Whether the small categories hold up. They do not, and not in the way I predicted.** I
expected documentation and enhancement to be confused with each other. They barely are: Mistral
never once mislabelled documentation as enhancement. The real pattern is that **`bug` leaks into
everything**, because it is 121 of the 253 test issues and a model that is unsure drifts towards
it. For Mistral the worst cells are bug→documentation (7), bug→enhancement (5) and bug→question
(5). Both models also collapse `other`: of its three test issues each model got one right and
answered `bug` for the others. The lesson is that the failure runs from the big class into the
small ones, rather than between the two small ones.

`other` is worth one more sentence, because it is where I would push back on my own headline
number. Three issues cannot support a per-class F1, and the harness marks the class `thin` for
that reason. Between my two test runs DeepSeek went from 0 of 3 to 1 of 3 on it, which alone moved
its macro-F1 by 0.067. Any six-class average on this answer key is partly a report on three
issues, and that is a limitation of my schema and my dataset size, not of the models.

**Whether writing out reasoning is worth the cost. No, and clearly.**
`deepseek-r1-distill-llama-70b` cost $0.00148 per call, 18 times the cheapest model, and scored
0.812 macro-F1 against the leader's 0.847. Its p95 latency was 115 seconds against a 120-second
timeout, which is why three calls failed outright. Choosing one word from a list of six does not
reward deliberation.

**Whether any model cannot follow the output format. Almost all of them can, and the one
apparent disaster was mine.** Nine of the eleven returned clean JSON on all 109 calls with no
fallback parsing. Only `nvidia-nemotron-3-super-120b` needed the last-resort text scan, twice.
`alibaba-qwen3-32b` initially failed to parse on 49.5% of calls, and that was my bug, not its
behaviour: I had catalogued it as a non-reasoning model, so it received a 96-token output budget,
spent all 96 thinking and never reached the JSON. Given room it produces valid JSON on 109 of 109
and scores 0.835. **On the original numbers I would have thrown out a competitive model for being
bad at a task it was never allowed to finish.**

### What I already know without that run

The findings below came out of building the test rather than running it, and two of them changed
how the test works.

**Accuracy on its own would have misled me, and I can demonstrate that.** I ran two deliberately
useless "models" against the real answer key. The first always answers `bug`, and it scored
**47.8% accuracy**. The second is twenty lines of text matching that answers `bug` unless the
issue title looks like a CVE report, and it scored **54.9% accuracy**. Both are useless, and the
second is better than chance while containing no model at all.

The macro-F1 score catches both of them, scoring them 0.108 and 0.280 respectively. It works by
scoring each of the six categories separately and then averaging those six scores equally, so a
model that falls back to the largest category whenever it is unsure is penalised for ignoring
the other five. My answer key contains 173 bugs and only 5 issues in the "other" category, so
this is a real weakness rather than a theoretical one. **That is why macro-F1 is my headline
number and accuracy is only a secondary figure.**

**Every issue in the security category was written by a bot.** All 26 issues labelled `security
vulnerability` are automated dependency-scanner reports, and they all take the same form:
`CVE-2020-9283 (High) detected in golang.org/x/crypto/ssh-…`. A simple text pattern identifies
them. This means that a model's score on the security category measures pattern matching rather
than any judgement about security, and every model will score close to full marks on one of the
six categories. I therefore flag these issues and report my headline score both with and without
them included. The text-matching model I described above falls from 0.280 to **0.136** once
these issues are removed, which is the clearest demonstration I have of how much they flatter
every model's score. In production I would filter them out with a text pattern and never pay a
model to look at them.

**No model can score near 100% on this task, and I can estimate roughly where the real ceiling
sits.** While building the answer key I set aside 15 issues as genuinely ambiguous, out of
roughly 100 that I examined closely. As an example, doctl issue #205 reports that the
`--volumes` option requires a UUID while every other option accepts names. That could
reasonably be called a bug, and it could reasonably be called a missing feature. Forcing it into
one category would not add information. It would add noise, which would then penalise whichever
model happened to read the issue the other way.

That refusal rate of roughly 15% implies that two careful people would disagree with each other
somewhere in the low teens as a percentage. It follows that **any model scoring in the low 90s
has reached the limit of what my data can measure, and a difference of one or two points between
two models means nothing at all.** This is the most useful thing to come out of the answer-key
work, and I would have lost it entirely if I had forced those 15 issues into categories.

**Using the maintainers' own labels alone would have broken the test quietly.** Those labels
were applied by different people over roughly ten years. The labels `suggestion` and
`enhancement` both exist and mean the same thing. The label `docs` was used **five times in the
repository's entire history**. An answer key built only from these labels would contain about
five documentation examples and no "other" examples at all, which would leave me unable to
measure whether a model can identify either category. The result would be a six-category test
that was really testing two categories.

For that reason the 362 answers come from two sources. **I translated 304 of them from the
maintainers' labels** using an explicit lookup table, and **I wrote 58 of them myself**,
deliberately looking for the categories that the maintainers' labels cannot supply. Every label
I wrote has a written reason recorded beside it.

**One category cannot honestly be measured.** The "other" category has only 3 issues in the
final scoring set, so a single issue moves its score by about 0.1. The application marks any
category with fewer than 10 examples as "thin". I would rather state plainly that a category
cannot be measured than invent examples to make the number look respectable.

### How I kept myself honest

I divided the 362 answers into two groups before I began testing anything. I kept 109 for
tuning and sealed 253 away for the final score, keeping the mix of categories similar in both
groups and using a fixed random seed so that the division can be reproduced.

**I wrote the prompt using only the tuning group. I chose the models using only the tuning
group. I looked at the sealed group once, at the end.** Had I screened all eleven models against
the sealed group and then reported the winner's score, that score would have been the best of
eleven noisy attempts, which would read considerably better than the truth.

| | bug | enhancement | question | security | documentation | other |
|---|---|---|---|---|---|---|
| tuning group | 52 | 34 | 8 | 8 | 5 | 2 |
| final scoring group | 121 | 78 | 20 | 18 | 13 | 3 |

Two bugs I found are worth admitting here, because each one would have given me wrong numbers
without producing any error message.

The prompt shows each model four worked examples, which are loaded by issue number, and the code
now checks that every one of them comes from the tuning group. In my first version those four
examples were typed in by hand, and **three of the four turned out to come from the sealed
group**. Had I not caught that, I would have been reporting the models' memory of examples they
had already seen as though it were accuracy.

The second bug concerned the pool of HTTP connections the application keeps open. That pool is
now sized according to the concurrency the current run is actually using. Previously it was
sized from the default setting, which meant that a run at concurrency 64 would have been limited
by a pool built for concurrency 8. Every latency figure would then have been measuring delays
inside my own code rather than the provider's response time.

Both problems are now covered by `make verify`, which runs just over forty checks. The exact
count depends on the provider, since one check only applies to the simulator.

A few other decisions are worth stating briefly. Every issue is sent as its own separate
request, and requests are never combined. Combining them would be cheaper, and the exercise
rules it out for four reasons I would have arrived at independently. You lose the timing for each
individual issue. You cannot retry one failed issue without repeating the whole group. One badly
formed reply can corrupt the issues grouped with it. And you cannot send a single difficult issue
to a stronger model, which is the mechanism my entire recommendation relies on. I also set
the temperature to 0 so that runs can be repeated exactly, and I use the same prompt for every
model, because varying the prompt would turn this into a comparison of prompts instead of models.

## How I would roll this out

This exercise proves something about a single repository. The customer's question concerns many
repositories, and I should say plainly that choosing the model is the easy part of that problem.

**Begin with a month of shadow mode.** Run the new model alongside whatever the customer uses
now. Record both sets of answers. Apply neither of them. Ship nothing to users.

Before going further I would want the level of agreement measured **for each product
separately**, not averaged across all of them. If the new model agrees with the current system
on 88% of doctl issues but only 61% of issues from a database product, then the categories mean
something different in that second product, and no amount of changing models will fix that. This
stage is not optional, and the reason is worth spelling out. doctl is a command-line tool, so
its issues are short, technical, written in English, and mostly written by developers. A managed
database product receives longer reports written by operations staff, containing more log output
and more English written by non-native speakers. **Nothing in my test measures that difference**,
and it is the most likely reason for a choice that looked good on doctl to perform badly
elsewhere.

**Build a separate answer key for each product** while shadow mode runs. I would aim for 150 to
250 issues per product, using the same two-source method I used here. This is where the real cost
of the project sits, and it is roughly one day of an engineer's attention per product rather
than money spent on computing. Before going further I would want two people to label the same
100 issues, so that there is a measured figure for how often people agree with each other. That
figure tells you when to stop improving the model. Pushing a score from 0.85 to 0.88 when two
people only agree with each other 0.86 of the time is spending money on noise.

**Then run a careful trial on one or two low-risk repositories.** Apply the model's label
automatically only when the model is confident. Below that level of confidence, obtain the second
opinion. If the two models still disagree, send the issue to a person. The confidence threshold
must be chosen from measured data rather than because 0.8 looks like a reasonable number, and the
confidence score itself must be validated first, because models are frequently confident and
wrong at the same time. Security issues should be treated differently from the rest: the system
should over-report them, and it should never close one automatically, because a wrongly
categorised issue is easy to correct while a missed vulnerability is not.

**Then widen the rollout** one product at a time, with the second opinion enabled. The rate at
which the two models disagree tells you directly how much work the second model will receive,
and therefore what that tier will cost.

**Where I expect this to break down.** Genuinely ambiguous issues are a real and permanent part
of the workload rather than a flaw in my labelling, so they should go to a person; the most
useful thing a classifier can report about a genuinely ambiguous issue is that it declines to
guess. Six categories will not survive contact with six products, because somebody will
eventually need a `billing` category, which is why the list of categories lives in one place and
the prompt is generated from that list rather than written out by hand for each category. The
way people write issues drifts over the years, so I would monitor how the spread of predicted
categories changes over time and raise an alarm if it drifts, because that check works without
needing correct answers. Models are eventually retired, and this harness is the regression test
for that event. Providers also update models without changing their names, so I would replay a
small fixed set of issues on a schedule and compare the results against the saved run for the
same dataset. Finally I would cap the length of model output and raise an alarm on the average
number of output tokens per call, because total spending only tells you about a problem after
the fact whereas the average output length tells you which model changed.

**I would not fine-tune a model yet.** I would only consider it once a straightforward prompted
model, together with the second opinion, has stopped improving and is still performing below the
level at which people agree with each other. That level is precisely the thing I have not
measured. Fine-tuning replaces a configuration change with a training pipeline, a problem of
versioning training data, and a model the customer then owns and must maintain.

## How to run it

```bash
docker build -t doctl-eval .
cp .env.example .env                 # put your key in DO_INFERENCE_API_KEY
docker run --rm -p 8080:8080 --env-file .env \
  -v "$PWD/data/runs:/app/data/runs" doctl-eval
# then open http://localhost:8080
```

The issues and the answer key are both built into the image, so a run needs neither access to
GitHub nor a GitHub token. Results are written to the folder you mounted.

To run it without Docker, use `make install && make verify`, then
`PROVIDER=digitalocean DO_INFERENCE_API_KEY=<key> make screen` to reproduce the model
comparison, and `make serve` to start the application. `make serve-mock` runs the simulator
offline and needs no key at all.

**Settings.** `DO_INFERENCE_API_KEY` holds the Serverless Inference API key, and it is the only
setting you must provide. It is never built into the image. Every other setting has a working
default.

| variable | default | what it does |
|---|---|---|
| `DO_INFERENCE_API_KEY` | — | **Required.** Your Serverless Inference API key. |
| `BASIC_AUTH_PASSWORD` | *(empty)* | **Set this before you deploy anywhere.** The login password. Leaving it empty removes the login entirely, which is only safe on localhost. |
| `BASIC_AUTH_USERNAME` | `reviewer` | The login username. |
| `PROVIDER` | `digitalocean` | Use `digitalocean` for real calls, or `mock` for the offline simulator. |
| `DO_INFERENCE_BASE_URL` | `https://inference.do-ai.run/v1` | Any endpoint that is compatible with the OpenAI API. |
| `CONCURRENCY` | `16` | How many requests run at the same time, shared between both models. 16 is measured rather than guessed: throughput rose almost in step with concurrency up to that point with no p95 penalty. It can also be changed per run in the application, so the image never needs rebuilding to change it. |
| `SCORED_SPLIT` | `test` | Use `test` for the reported score, or `dev` while tuning. |
| `MAX_ISSUES` | `0` | `0` means all 536 issues. A number above 0 takes an evenly spread sample, which is useful for quick tests. |
| `REASONING_MAX_TOKENS` | `1400` | How much room to allow models that write out their reasoning. Setting it too low produces `parse_error` results. |
| `REQUEST_TIMEOUT_S` / `MAX_RETRIES` / `TEMPERATURE` | `120` / `3` / `0` | How long to wait for a reply, how many attempts to make, and how much randomness to allow. 120 seconds because two models in the catalog have a p95 above one minute, and at 60 they fail as timeouts rather than reporting as slow. |
| `MAX_TOKENS` | `96` | Output cap for models that answer with bare JSON. Too low and replies are cut off mid-JSON, which shows up as `parse_error` rather than as anything mentioning tokens. |
| `PORT` | `8080` | The port the server listens on. |

All the settings are documented with their defaults in [`.env.example`](.env.example).

**A note on the login.** The application sits behind HTTP Basic auth whenever
`BASIC_AUTH_PASSWORD` is set. The reason is money rather than privacy: `POST /api/run` spends
real credits using the key in the container's environment, a full run costs roughly $0.27, and
there is no rate limit. An unprotected public URL could therefore drain the credit balance in
an afternoon. If you leave the password empty every route is open, which is convenient on a
laptop and unsafe anywhere else, so set it before you deploy. `/api/health` stays open either
way so that a platform health check can reach it without credentials.

I should be clear about what this is not. Basic auth gives one shared username and password. It
provides no user accounts, no per-caller spending limit, and no record of who ran what. For a
single-user evaluation tool served over HTTPS that is the right amount of security. For
anything with several users it would not be.

## The data files

- `data/corpus/doctl-issues-snapshot.json` holds the frozen set of 536 issues, with the
  fingerprint `18d67e20321158c9`.
- `data/ground_truth/gold.json` holds the 362 answers, each recording where it came from and why.
- `data/ground_truth/hand_labels.json` holds the 58 answers I wrote myself, along with the 15
  issues I declined to answer.
- `data/ground_truth/ANNOTATION_GUIDE.md` holds the labelling rules, which I wrote before I
  wrote any labels.
- `data/runs/` holds every completed run, with the full detail for every issue: the category
  chosen, the raw reply, the timing, the token counts and the cost breakdown.
- `data/screening/` holds the tables comparing all eleven models.

Two notes on those last two folders. The screening file currently in the repository was produced
by the simulator, which is why its name contains `sim` rather than `live`; the real one appears
once `make screen` has been run against an API key.

And on the deployed copy, `data/runs/` empties whenever the app restarts or redeploys. App
Platform gives a container a temporary filesystem rather than a persistent disk, so saved runs do
not survive. That is worth knowing before you conclude the run list is broken. Runs made on a
local machine persist normally. Keeping them on the server would mean writing to Spaces or a
database, which is more machinery than an evaluation harness needs.

## What I did not do, and why

**I did not compare against a frontier model.** The credits do not cover Claude or GPT-5.x, so
the question of whether the customer is overpaying is answered inside the range the credits
reach. I had intended `qwen3.5-397b-a17b` to represent the top of that range, and it rate-limited
105 of 109 calls, so I have no stand-in for the expensive tier either. What I can say is that
across the ten models I *could* measure, spanning 14B to 284B and an 18x range of cost per call, quality
differences were inside run-to-run noise. That does not prove a frontier model would add nothing.
It does mean nothing in this field showed a size-related advantage worth paying for.

**I did not have a second person check the answer key.** One person wrote it in a single pass,
so I have no measurement of how often two people would agree. This is the answer key's greatest
weakness and the first thing I would correct at a larger scale.

**I did not check whether the confidence scores mean anything.** I record them for every call
but I never verified that they separate correct answers from wrong ones. Until somebody does
verify that, my recommendation routes work based on the two models disagreeing, which is
something I have measured.

**I did not fine-tune, use retrieval, or combine several models by voting.** All three are
reasonable next steps, and all three are premature while a straightforward prompted model has
not yet stopped improving against a ceiling that nobody has measured.

**I did not label all 536 issues by hand.** 174 issues have no correct answer recorded and
appear in the unscored view of the application instead. A real customer's workload is mostly
unlabelled, so that part of the application does useful work rather than sitting unused.

**I did not use scikit-learn.** Precision, recall and F1 across six categories takes about
twenty lines to write, and the exercise asks for the arithmetic to be checkable. You can read my
definition of F1 in `metrics.py`. You cannot read one that sits inside an imported library.
