## Overview

Apply the configured `head_dtype` to BERT masked-language-model and SPLADE heads instead of silently constructing those heads in the encoder's default dtype.

## Claims

- Pass the resolved head dtype to the MLM transform, LayerNorm, and vocabulary decoder.
- Convert hidden states to the live transform weight dtype before applying the MLM head.
- Preserve the default dtype behavior when `BertMLMHead` is constructed directly without an explicit dtype.

## Validation

- [Native before/after comparison](https://github.com/Afloat16/vllm/actions/runs/43424033592): **6 failures / 8 passes on the original source**, then **14 passes with this change**.
- [Final exact-source native validation](https://github.com/Afloat16/vllm/actions/runs/43426040940): **14 new tests and 76 existing pooler tests pass**.
- [Repository pre-commit hooks and Ruff](https://github.com/Afloat16/vllm/actions/runs/43424810234) pass.
- The tests save and load real small BERT checkpoints, then call public `LLM.encode` for both MLM token classification and SPLADE sparse embeddings.
- FP32 and BF16 encoders are combined with default, explicit FP32, `model`, and explicit BF16 head settings. Exact output references and native worker parameter-dtype inspection distinguish the head precision from the encoder precision.

```bash
pytest tests/models/language/pooling/test_head_dtype.py -k mlm_head_dtype
```

## Details

The model configuration already resolves `head_dtype`, with FP32 as the default, but these native MLM heads were constructed without it. A BF16 encoder therefore also rounded the MLM weights and bias to BF16, despite a configured FP32 head. Explicit lower-precision heads likewise did not follow the requested setting.

The saved-checkpoint tests use exact bias values and an independent dtype-rounding reference to distinguish encoder precision from head precision, as well as inspecting the loaded parameter dtypes. They exercise both model architectures and their actual public pooling APIs. This change does not alter checkpoint alias handling or claim to recover precision already lost through an existing tied-weight fallback.

The official C++ and Rust CPU runtime was compiled from the unchanged base `191e5d5` and reused for the Python-only change. Native execution was on CPU; no GPU or multi-rank execution is claimed.

The final commit carrying the DCO sign-off has the identical Git tree as validated commit `a3af37f3e544a4434a52f75414731a10ae176303`; only commit history and the sign-off changed.
