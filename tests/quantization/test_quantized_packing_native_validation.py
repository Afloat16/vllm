# Validation-only end-to-end checkpoint packing comparison.
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_cols,
    pack_quantized_values_into_int32,
)
from vllm.sampling_params import SamplingParams
from vllm.scalar_type import scalar_types


def save_quantized_checkpoint(path, storage_dtype):
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
        if name.startswith("model.layers.") and name.endswith(".weight") and value.ndim == 2:
            prefix = name.removesuffix(".weight")
            value = value.t().contiguous()
            k, n = value.shape
            grouped = value.reshape(k // 32, 32, n)
            scales = (grouped.abs().amax(dim=1).clamp_min(1e-6) / 7).half()
            codes = (grouped / scales.float().unsqueeze(1)).round().clamp(-8, 7) + 8
            codes = codes.reshape(k, n).to(storage_dtype)
            weights[prefix + ".qweight"] = pack_quantized_values_into_int32(
                codes, scalar_types.uint4b8, 0
            ).contiguous()
            weights[prefix + ".scales"] = scales
            weights[prefix + ".qzeros"] = pack_cols(
                torch.full((k // 32, n), 7, dtype=torch.int32), 4, k // 32, n
            )
            weights[prefix + ".g_idx"] = torch.arange(k, dtype=torch.int32) // 32
        else:
            weights[name] = value.half().contiguous()
    model.config.quantization_config = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
    }
    model.config.save_pretrained(path)
    save_file(weights, str(path / "model.safetensors"), metadata={"format": "pt"})


@pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.int8, torch.int16])
def test_narrow_weight_packing_llm_outputs(tmp_path, vllm_runner, storage_dtype):
    reference_path = tmp_path / "reference"
    candidate_path = tmp_path / "candidate"
    save_quantized_checkpoint(reference_path, torch.int32)
    save_quantized_checkpoint(candidate_path, storage_dtype)
    results = []
    for path in (reference_path, candidate_path):
        with vllm_runner(
            str(path),
            dtype="float16",
            quantization="auto_gptq",
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
                [{"prompt_token_ids": [1, 7, 12]}, {"prompt_token_ids": [1, 5, 19, 28]}],
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
            max_error = max(max_error, max(abs(elp[t].logprob - alp[t].logprob) for t in elp))
    print(json.dumps({"storage_dtype": str(storage_dtype), "token_mismatches": mismatches, "max_logprob_error": max_error}), flush=True)
    assert not mismatches
    assert max_error <= 1e-6
