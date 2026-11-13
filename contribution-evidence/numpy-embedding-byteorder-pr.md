## Overview

Allow image and video embedding readers to consume NumPy `.npy` arrays whose byte order differs from the host, preserving their values and dtype.

## Claims

- Normalize a non-native NumPy dtype to native byte order before constructing a Torch tensor.
- Use the same conversion for image/video bytes, base64, and file loading.
- Keep the native-endian `torch.from_numpy` path and the existing `allow_pickle=False` behavior.

## Validation

- [Native CPU validation](https://github.com/Afloat16/vllm/actions/runs/43423892421): **27 failures / 23 passes on the original source**, then **50 passes with this change**.
- **31 existing serialization and embedding tests pass.**
- [Repository pre-commit hooks and Ruff](https://github.com/Afloat16/vllm/actions/runs/43423896475) pass.
- Both native-endian and big-endian input arrays are tested through the real bytes, base64, and file readers. Floating-point values and large int64 metadata are compared with independent Python-value references.
- Two saved tiny Llava cases run the actual public `LLM.generate` path. The directly constructed and decoded image embeddings produce matching token IDs and all 32 vocabulary log probabilities.

```bash
pytest tests/renderers/test_numpy_embedding_byteorder.py
```

## Details

A `.npy` file carries its dtype and byte order. `np.load` correctly reads a big-endian array, but passing that array directly to `torch.from_numpy` raises an unsupported-byte-order error on a little-endian host. Re-encoding with `astype(dtype.newbyteorder('='))` converts the numerical representation instead of merely relabeling the bytes.

The regression cases cover FP16/FP32/FP64, transposed/strided inputs, empty shapes, large int64 values, and object-array rejection. A saved tiny Llava model consumes an image embedding loaded through the real base64 handler and produces the same tokens and logprobs as the equivalent directly constructed tensor. Its test checkpoint uses the CLIP wrapper names supported by the native loader; the tensors and model execution are real.

The official C++ and Rust CPU runtime was built from base `191e5d5` and reused for this Python-only change. The tests run on CPU; they do not claim GPU coverage or a model-quality increase. Native-endian arrays keep the existing `torch.from_numpy` path. Foreign-endian arrays require one value-preserving conversion.

The signed commit has the identical Git tree as validated commit `50abe3b5a09cd3e505762c92f2fceccf3ebdd38d`; only commit history and the sign-off changed.
