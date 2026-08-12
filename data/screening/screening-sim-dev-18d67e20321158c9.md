# Model screening — dev split, 6 issues

- provider: `mock`  **(SIMULATED — not evidence)**
- corpus: `18d67e20321158c9`
- prompt: `v3`
- concurrency: 8
- rates: https://docs.digitalocean.com/products/inference/details/pricing/ (verified 2026-08-07)

| model | params | arch | macro-F1 | macro-F1 excl. templated | accuracy | p50 ms | p95 ms | mean out tok | $/call | $/correct | err % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mistral-3-14B` | 14B | dense | 0.889 | 0.833 | 83.3% | 426 | 875 | 22 | $0.000267 | $0.00032 | 0.0% |
