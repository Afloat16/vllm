import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast


def run_case(directory, dim_mode):
    from vllm import LLM

    torch.manual_seed(19)
    directory = Path(directory)
    config = BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=64, pad_token_id=0)
    model = BertModel(config, add_pooling_layer=False).eval()
    projection = torch.randn(8, 16) * 0.1
    config.architectures = ["ColBERTModel"]
    if dim_mode != "inferred":
        setattr(config, dim_mode, 8)
    config.save_pretrained(directory)
    state = {"bert." + key: value.contiguous() for key, value in model.state_dict().items()}
    state["linear.weight"] = projection.contiguous()
    save_file(state, directory / "model.safetensors", metadata={"format": "pt"})
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                  "alpha": 4, "beta": 5, "gamma": 6}
    vocabulary.update({f"unused{i}": i for i in range(7, 16)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
    hf_tokenizer.save_pretrained(directory)
    llm = LLM(model=str(directory), runner="pooling", dtype="float32",
              max_model_len=64, max_num_batched_tokens=128, max_num_seqs=2,
              kv_cache_memory_bytes=16 * 1024 * 1024,
              enforce_eager=True, enable_prefix_caching=False, seed=0)
    texts = ["alpha beta", "gamma"]
    outputs = llm.encode(texts, pooling_task="token_embed", use_tqdm=False)
    expected = []
    with torch.inference_mode():
        for text in texts:
            inputs = hf_tokenizer(text, return_tensors="pt")
            hidden = model(**inputs).last_hidden_state[0]
            expected.append(F.normalize(F.linear(hidden, projection), p=2, dim=-1))
    differences = []
    for actual, reference in zip(outputs, expected):
        tensor = actual.outputs.data
        torch.testing.assert_close(tensor, reference, atol=1e-5, rtol=1e-5)
        differences.append(float((tensor - reference).abs().max()))
    scores = llm.score(texts[0], texts[1], use_tqdm=False)
    reference_score = (expected[0] @ expected[1].T).amax(-1).sum()
    torch.testing.assert_close(torch.tensor(scores[0].outputs.score), reference_score,
                               atol=1e-5, rtol=1e-5)
    print("NATIVE_COLBERT_RESULT " + json.dumps({"mode": dim_mode,
          "shapes": [list(out.outputs.data.shape) for out in outputs],
          "embedding_max_errors": differences, "score": scores[0].outputs.score}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_case(sys.argv[1], sys.argv[2])
    else:
        records = []
        with tempfile.TemporaryDirectory() as root:
            for mode in ("inferred", "colbert_dim", "dim", "projection_dim"):
                directory = Path(root) / mode
                directory.mkdir()
                completed = subprocess.run(
                    [sys.executable, __file__, str(directory), mode],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=360)
                Path(f"probe_colbert_{mode}.log").write_text(completed.stdout)
                print(f"=== NATIVE COLBERT {mode}: exit {completed.returncode} ===", flush=True)
                print(completed.stdout[-18000:], flush=True)
                records.append({"mode": mode, "returncode": completed.returncode})
        Path("probe_pooling_results.json").write_text(json.dumps(records, indent=2))
