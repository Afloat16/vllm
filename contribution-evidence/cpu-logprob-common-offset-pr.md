## Overview

Prevent cancellation from corrupting token log probabilities in the standalone CPU model runner when logits share a large offset.

## Claims

- Subtract the row maximum before computing both the selected logit and normalization term.
- Preserve the existing FP32 output and nonfinite-row behavior.
- Keep top-k and prompt-score callers on the same corrected token-logprob implementation.

## Validation

- [Exact-source native ARM CPU validation](https://github.com/Afloat16/vllm/actions/runs/43419952030): **19 failures / 9 passes on the original source**, then **28 passes with this change**.
- **10 existing CPU logprob tests pass.**
- [Repository pre-commit hooks and Ruff](https://github.com/Afloat16/vllm/actions/runs/43425207334) pass.
- Tests cover FP16/BF16/FP32 logits, positive and negative common offsets, masked and nonfinite rows, empty selected-token sets, top-k scores, and prompt scores.
- Six saved tiny GPT-2 cases exercise the real public `LLM.generate` path in raw and processed logprob modes. The native C++ and Rust runtime was built for the unchanged base `191e5d5` on ARM and reused for the exact final-source run.

```bash
pytest tests/v1/worker/cpu/test_cpu_logprob.py
```

## Details

Computing a large log-normalizer and then subtracting it from a similarly large selected logit discards the small log probability. For example, a real saved tiny GPT-2 with sixteen equal logits of `1e8` returned `0` for each token instead of `-log(16)` (approximately `-2.7725887`). The same failure occurs with a large negative common offset.

The new calculation keeps the common offset out of both terms. This change targets `vllm/v1/worker/cpu/logprob.py`, used by the standalone CPU runner. The x86 Triton runner uses a different implementation; the new end-to-end tests therefore run on native ARM CPU hardware. No GPU path or model-quality improvement is claimed.

Native validation used commit `8b376970f56d33e8a543ee90fa69978cffe0d5d0`. The code commit carrying the DCO sign-off `a1d104587bcf4b86592fe988ac05f4b6cd47ede4` has the identical Git tree. Subsequent commits only add and clarify rendered user documentation; the production Python file and tests remain identical to the native-tested code.
