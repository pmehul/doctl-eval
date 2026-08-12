# Model screening — dev split, 109 issues

- provider: `digitalocean`
- corpus: `18d67e20321158c9`
- prompt: `v3`
- concurrency: 32
- rates: https://docs.digitalocean.com/products/inference/details/pricing/ (verified 2026-08-07)

| model | params | arch | macro-F1 | macro-F1 excl. templated | accuracy | p50 ms | p95 ms | rps | mean out tok | $/call | $/correct | err % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-4-flash` | 284B total | moe | 0.800 | 0.759 | 82.6% | 3196 | 4308 | 1.70 | 16 | $8.14e-05 | $9.86e-05 | 0.0% |
