# Model screening — dev split, 109 issues

- provider: `mock`  **(SIMULATED — not evidence)**
- corpus: `18d67e20321158c9`
- prompt: `v3`
- concurrency: 8
- rates: https://docs.digitalocean.com/products/inference/details/pricing/ (verified 2026-08-07)

| model | params | arch | macro-F1 | macro-F1 excl. templated | accuracy | p50 ms | p95 ms | mean out tok | $/call | $/correct | err % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `openai-gpt-oss-120b` | ~117B total / ~5.1B active | moe | 0.835 | 0.802 | 85.3% | 935 | 1694 | 22 | $0.000143 | $0.000168 | 0.0% |
| `qwen3.5-397b-a17b` | 397B total / 17B active | moe | 0.791 | 0.767 | 83.3% | 1665 | 3884 | 22 | $0.000429 | $0.000519 | 0.9% |
| `alibaba-qwen3-32b` | 32.8B | dense | 0.757 | 0.708 | 81.7% | 873 | 1535 | 22 | $0.000332 | $0.000407 | 0.0% |
| `deepseek-4-flash` | 284B total | moe | 0.743 | 0.691 | 80.7% | 913 | 2012 | 22 | $9.07e-05 | $0.000112 | 0.0% |
| `llama-4-maverick` | 400B total / 17B active | moe | 0.733 | 0.701 | 80.4% | 1112 | 2177 | 23 | $0.000272 | $0.000344 | 1.8% |
| `deepseek-r1-distill-llama-70b` | 70B | dense · reasoning | 0.714 | 0.700 | 80.4% | 5408 | 14442 | 703 | $0.00196 | $0.00249 | 1.8% |
| `nvidia-nemotron-3-super-120b` | 120B | dense | 0.685 | 0.654 | 80.4% | 1072 | 1923 | 22 | $0.000219 | $0.000277 | 1.8% |
| `mistral-3-14B` | 14B | dense | 0.658 | 0.612 | 76.1% | 527 | 1016 | 22 | $0.00026 | $0.000342 | 0.0% |
| `gemma-4-31B-it` | 31B | dense | 0.631 | 0.640 | 71.6% | 783 | 1477 | 21 | $0.000241 | $0.000337 | 0.0% |
| `llama3.3-70b-instruct` | 70B | dense | 0.620 | 0.590 | 78.0% | 1300 | 2834 | 22 | $0.000846 | $0.00108 | 0.0% |
| `openai-gpt-oss-20b` | ~21B total / ~3.6B active | moe | 0.500 | 0.531 | 60.6% | 667 | 1368 | 22 | $7.4e-05 | $0.000122 | 0.0% |
