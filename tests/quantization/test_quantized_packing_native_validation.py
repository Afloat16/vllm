# Validation-only: preserve dequantized weights through real native CPU inference.
# This host has AVX2, so the WNA16 kernel is unavailable. No substitute kernel is
# registered: unpacked weights are saved as an ordinary FP32 Llama checkpoint.
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM
from vllm.sampling_params import SamplingParams
from vllm.scalar_type import scalar_types

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)


@pytest.mark.parametrize(
    "storage_dtype", [torch.uint8, torch.int8, torch.int16, torch.int32]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_weights_preserve_torch_linear_outputs(storage_dtype, dtype):
    torch.manual_seed(31)
    k, n, group_size = 64, 64, 32
    codes = torch.randint(0, 16, (k, n), dtype=storage_dtype)
    packed = pack_quantized_values_into_int32(codes, scalar_types.uint4b8, 1)
    restored = unpack_quantized_values_into_int32(packed, scalar_types.uint4b8, 1)
    scales = (torch.rand(k // group_size, n) / 16).to(dtype)
    scale_rows = scales.repeat_interleave(group_size, dim=0)
    expected_weight = (codes.to(torch.int32) - 8).to(dtype) * scale_rows
    actual_weight = (restored - 8).to(dtype) * scale_rows
    inputs = torch.randn(2, 3, k, dtype=dtype)
    expected = torch.nn.functional.linear(inputs, expected_weight.t())
    actual = torch.nn.functional.linear(inputs, actual_weight.t())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def save_dequantized_checkpoint(path, storage_dtype):
    path.mkdir()
    torch.manual_seed(713)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    ).eval()
    weights = {}
    for name, value in model.state_dict().items():
        if (
            name.startswith("model.layers.")
            and name.endswith(".weight")
            and value.ndim == 2
        ):
            value = value.t().contiguous()
            k, n = value.shape
            grouped = value.reshape(k // 32, 32, n)
            scales = grouped.abs().amax(dim=1).clamp_min(1e-6) / 7
            codes = ((grouped / scales.unsqueeze(1)).round().clamp(-8, 7) + 8).reshape(
                k, n
            )
            if storage_dtype is None:
                # Independent expected weights do not use either packing helper.
                restored = codes
            else:
                packed = pack_quantized_values_into_int32(
                    codes.to(storage_dtype), scalar_types.uint4b8, 0
                )
                restored = unpack_quantized_values_into_int32(
                    packed, scalar_types.uint4b8, 0
                ).float()
            value = (
                ((restored.reshape(k // 32, 32, n) - 8) * scales.unsqueeze(1))
                .reshape(k, n)
                .t()
            )
        weights[name] = value.contiguous()
    model.config.save_pretrained(path)
    save_file(weights, str(path / "model.safetensors"), metadata={"format": "pt"})


@pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.int8, torch.int16])
def test_narrow_weight_packing_llm_outputs(tmp_path, vllm_runner, storage_dtype):
    reference_path = tmp_path / "reference"
    candidate_path = tmp_path / "candidate"
    save_dequantized_checkpoint(reference_path, None)
    save_dequantized_checkpoint(candidate_path, storage_dtype)
    results = []
    for path in (reference_path, candidate_path):
        with vllm_runner(
            str(path),
            dtype="float32",
            skip_tokenizer_init=True,
            max_model_len=32,
            max_num_seqs=1,
            async_scheduling=False,
            enforce_eager=True,
            enable_prefix_caching=False,
            kv_cache_memory_bytes=16 * 1024 * 1024,
            max_logprobs=64,
        ) as runner:
            result = runner.llm.generate(
                [
                    {"prompt_token_ids": [1, 7, 12]},
                    {"prompt_token_ids": [1, 5, 19, 28]},
                ],
                SamplingParams(
                    temperature=0,
                    max_tokens=4,
                    logprobs=64,
                    ignore_eos=True,
                    detokenize=False,
                ),
                use_tqdm=False,
            )
            results.append(result)
    mismatches = []
    max_error = 0.0
    for expected, actual in zip(results[0], results[1]):
        e, a = expected.outputs[0], actual.outputs[0]
        if a.token_ids != e.token_ids:
            mismatches.append({"expected": e.token_ids, "actual": a.token_ids})
        for elp, alp in zip(e.logprobs, a.logprobs):
            assert set(elp) == set(alp)
            max_error = max(
                max_error, max(abs(elp[t].logprob - alp[t].logprob) for t in elp)
            )
    print(
        json.dumps(
            {
                "storage_dtype": str(storage_dtype),
                "token_mismatches": mismatches,
                "max_logprob_error": max_error,
            }
        ),
        flush=True,
    )
    assert not mismatches
    assert max_error <= 1e-6
