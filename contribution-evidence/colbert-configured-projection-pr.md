## Overview

Fix loading ColBERT checkpoints that declare their projection width through `colbert_dim`, `dim`, or `projection_dim`.

## Claims

- Construct the projection after `nn.Module` initialization and before the token pooler consumes it.
- Accept both existing projection checkpoint names, `linear.weight` and `colbert_linear.weight`, with configured or inferred dimensions.
- Record both registered names of the shared projection after its weight is actually loaded.

## Validation

- [Native CPU validation](https://github.com/Afloat16/vllm/actions/runs/43415980870): the same eight new saved-checkpoint cases produce **6 failures / 2 passes on the original source**, then **8 passes with this change**.
- **76 existing pooler head and activation tests pass.**
- [Repository pre-commit hooks and Ruff](https://github.com/Afloat16/vllm/actions/runs/43412052619) pass.
- Each new case saves a local one-layer BERT plus a non-square projection, loads it through the public `LLM` API, and compares token embeddings and MaxSim scores with the actual Transformers encoder and an independent projection reference.
- Validation used the compiled C++/Rust CPU runtime for base `191e5d5`, with PyTorch 2.14. No GPU or distributed-model validation is claimed.

Reproduce the new tests with:

```bash
pytest tests/models/language/pooling/test_colbert.py -k configured_projection
```

## Details

The child constructor initialized the ColBERT projection before the parent initialized `nn.Module`. With an explicit projection dimension, assigning the `Linear` submodule therefore raised `AttributeError` during model startup. Moving that initialization into the existing model-building hook preserves the parent initialization order and makes the projection available to the pooler.

The projection is also registered through the pooler. Its successful checkpoint load now reports both aliases, so the loaded-parameter check agrees with the actual shared module. The inference-dimension path remains covered.

The final commit carrying the DCO sign-off has exactly the same Git tree as validated commit `edb858436acebd612073a859cfd3ea6bb5cf2a48`; only commit history and the sign-off changed.
