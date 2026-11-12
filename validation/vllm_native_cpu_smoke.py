import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def main():
    torch.manual_seed(74)
    model_dir = Path("validation/local-gpt2")
    model_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"word{i}": i + 4 for i in range(28)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>",
    )
    fast.save_pretrained(model_dir)
    config = GPT2Config(
        vocab_size=32, n_embd=128, n_layer=2, n_head=2,
        n_positions=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )
    reference = GPT2LMHeadModel(config).eval()
    reference.save_pretrained(model_dir)
    prompts = [[1, 5, 6], [1, 8, 9]]
    expected = []
    for prompt in prompts:
        tokens = torch.tensor([prompt])
        with torch.no_grad():
            for _ in range(4):
                next_token = reference(tokens).logits[:, -1].argmax(-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
        expected.append(tokens[0, len(prompt):].tolist())

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=str(model_dir), tokenizer=str(model_dir), dtype="float32",
        enforce_eager=True, max_model_len=64, max_num_seqs=2,
        enable_prefix_caching=False, seed=74,
    )
    results = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=3),
        use_tqdm=False,
    )
    actual = [list(result.outputs[0].token_ids) for result in results]
    assert actual == expected, (actual, expected)
    print("NATIVE_CPU_SMOKE=" + json.dumps({
        "parameters": sum(p.numel() for p in reference.parameters()),
        "torch": torch.__version__, "prompts": prompts,
        "reference_tokens": expected, "vllm_tokens": actual,
    }))


if __name__ == "__main__":
    main()
