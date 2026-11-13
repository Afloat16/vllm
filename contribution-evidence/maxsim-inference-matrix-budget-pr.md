## Overview

Keep the reference MaxSim similarity matrix within its element budget during inference, including when a single query/document pair is larger than the entire budget.

## Claims

- Tile oversized single pairs along query and document tokens while preserving FP32 MaxSim scores.
- Keep each temporary similarity matrix within `max_score_matrix_elements`.
- Preserve the existing full-matrix path when input gradients are tracked.

## Validation

- [Native CPU tests and benchmark](https://github.com/Afloat16/vllm/actions/runs/43417368801): **2 allocation failures / 20 passes on the original source**, then **22 passes with the change**.
- **9 existing reference-runner tests pass; 2 existing CUDA-only cases are skipped on the CPU host.**
- [Repository pre-commit hooks and Ruff](https://github.com/Afloat16/vllm/actions/runs/43417250570) pass.
- Regression cases cover FP16/BF16/FP32 inputs, strided tensors, different budgets, negative scores, input ownership, allocation bounds, and the unchanged gradient-tracked path.
- A real saved tiny ColBERT model exercises the worker-side scoring path with an 8,193-token query and document. Its score agrees with independently obtained token embeddings.

On an AMD EPYC 7763 CPU host, two Torch threads, three fresh processes per configuration, and seven timed calls per process:

| Query × document tokens | Budget | Original median | Patched median | Original additional peak RSS | Patched additional peak RSS |
| --- | --- | --- | --- | --- | --- |
| 32 × 512 | 1,048,576 elements | 0.07358 ms | 0.07326 ms | 5.574 MiB | 5.574 MiB |
| 4,096 × 8,192 | 1,048,576 elements | 48.814 ms | 9.160 ms | 134.008 MiB | 9.684 MiB |

The benchmark uses 16-dimensional embeddings and produces identical scalar scores. These are local reference-scorer measurements, not full-model or GPU throughput claims. The native validation branch contains the benchmark script and per-process results.

```bash
pytest tests/v1/worker/test_late_interaction_runner.py
pytest tests/models/language/pooling/test_colbert.py -k long_pair_worker
```

## Details

Reducing the number of pairs in a batch cannot shrink the matrix once only one pair remains. The new inference fallback computes document maxima across bounded tiles, concatenates the per-query maxima, and sums them. Ordinary pairs retain the existing batched implementation.

The budget covers the temporary similarity matrix, not model weights, token embeddings, attention memory, or the fused GPU implementation. Gradient-tracked calls keep the original path; this change does not repair the existing backward limitation caused by the original in-place masking.

The final commit carrying the DCO sign-off has exactly the same Git tree as validated commit `ebdd3f7a92c51731a07adc5a3e90a5af6e0ba3b4`; only commit history and the sign-off changed.
