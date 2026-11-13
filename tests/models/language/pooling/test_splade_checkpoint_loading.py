# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.parametrize(
    "checkpoint_layout",
    [
        "hf_safetensors",
        "hf_sharded",
        "explicit_aliases",
        "decoder_names",
        "canonical_bias",
    ],
)
def test_splade_checkpoint_aliases(
    tmp_path, vllm_runner, tie_word_embeddings, checkpoint_layout
):
    """Load saved HF MLM weights and compare actual sparse embedding outputs."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertForMaskedLM, TokenizersBackend

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        config = BertConfig(
            vocab_size=16,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=64,
            pad_token_id=0,
            cls_token_id=2,
            sep_token_id=3,
            tie_word_embeddings=tie_word_embeddings,
        )
        reference = BertForMaskedLM(config).eval()
        with torch.no_grad():
            reference.cls.predictions.bias.copy_(torch.linspace(-0.4, 0.7, 16))

    if checkpoint_layout.startswith("hf_"):
        reference.save_pretrained(
            tmp_path,
            max_shard_size="2KB" if checkpoint_layout == "hf_sharded" else "5GB",
        )
    else:
        config.save_pretrained(tmp_path)
        state = {
            name: value.detach().clone().contiguous()
            for name, value in reference.state_dict().items()
        }
        if checkpoint_layout == "decoder_names":
            del state["cls.predictions.bias"]
        elif checkpoint_layout == "canonical_bias":
            del state["cls.predictions.decoder.bias"]
        save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})

    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
    }
    vocabulary.update({f"unused{i}": i for i in range(7, 16)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(tmp_path)
    texts = ["alpha", "gamma beta alpha", "beta beta"]
    expected = []
    with torch.no_grad():
        for text in texts:
            logits = reference(**tokenizer(text, return_tensors="pt")).logits[0]
            expected.append(torch.log1p(torch.relu(logits[1:-1])).amax(dim=0))

    with vllm_runner(
        str(tmp_path),
        runner="pooling",
        dtype="float32",
        hf_overrides={"architectures": ["BertSpladeSparseEmbeddingModel"]},
        max_model_len=64,
        max_num_batched_tokens=128,
        max_num_seqs=3,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        enable_prefix_caching=False,
        enforce_eager=True,
    ) as model:
        actual = model.embed(texts)

    assert len(actual) == len(expected)
    for vector, target in zip(actual, expected):
        assert len(vector) == config.vocab_size
        torch.testing.assert_close(
            torch.as_tensor(vector), target, rtol=1e-5, atol=1e-5
        )
