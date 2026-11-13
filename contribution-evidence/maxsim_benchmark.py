import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

BASE = "191e5d5df2ec693d5536a635079d58864dfacebb"
SOURCE = Path("vllm/v1/pool/late_interaction.py")

def child(query_len, doc_len):
    import torch
    from vllm.v1.pool.late_interaction import compute_maxsim_score_batched
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(71)
    query = torch.randn(query_len, 16, generator=generator)
    document = torch.randn(doc_len, 16, generator=generator)
    budget = 1_048_576
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.no_grad():
        score = compute_maxsim_score_batched([query], [document], max_score_matrix_elements=budget)[0]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            observed = compute_maxsim_score_batched([query], [document], max_score_matrix_elements=budget)[0]
            timings.append(time.perf_counter() - start)
            torch.testing.assert_close(observed, score)
    print("BENCH_RESULT=" + json.dumps({
        "query_len": query_len, "doc_len": doc_len, "dimension": 16, "budget": budget,
        "score": score.item(), "median_ms": statistics.median(timings) * 1000,
        "peak_rss_kib": peak, "incremental_peak_kib": max(0, peak - before),
    }))

def parent():
    fixed = SOURCE.read_bytes()
    records = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        for mode in ("before", "after"):
            SOURCE.write_bytes(
                subprocess.check_output(["git", "show", BASE + ":" + str(SOURCE)])
                if mode == "before" else fixed
            )
            for query_len, doc_len in [(32, 512), (4096, 8192)]:
                for repeat in range(3):
                    process = subprocess.run(
                        [sys.executable, "-B", __file__, "--child", str(query_len), str(doc_len)],
                        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
                    )
                    Path(f"maxsim_benchmark_{mode}_{query_len}_{repeat}.log").write_text(process.stdout)
                    line = [s for s in process.stdout.splitlines() if s.startswith("BENCH_RESULT=")][-1]
                    record = json.loads(line.split("=", 1)[1])
                    record.update(mode=mode, repeat=repeat)
                    records.append(record)
                    print(json.dumps(record), flush=True)
    finally:
        SOURCE.write_bytes(fixed)
    Path("maxsim_benchmark.json").write_text(json.dumps(records, indent=2) + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(int(sys.argv[2]), int(sys.argv[3]))
    else:
        parent()
