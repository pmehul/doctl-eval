# Model screening — dev split, 109 issues

- provider: `digitalocean`
- corpus: `18d67e20321158c9`
- prompt: `v3`
- concurrency: 16
- rates: https://docs.digitalocean.com/products/inference/details/pricing/ (verified 2026-08-07)

| model | params | arch | macro-F1 | macro-F1 excl. templated | accuracy | p50 ms | p95 ms | rps | mean out tok | $/call | $/correct | err % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mistral-3-14B` | 14B | dense | 0.816 | 0.779 | 86.2% | 1352 | 3104 | 9.16 | 17 | $0.000239 | $0.000277 | 0.0% |
| `deepseek-4-flash` | 284B total | moe | 0.779 | 0.734 | 83.5% | 2438 | 5470 | 4.96 | 16 | $8.14e-05 | $9.75e-05 | 0.0% |
