import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = "191e5d5df2ec693d5536a635079d58864dfacebb"
SOURCE = Path("vllm/v1/worker/cpu/logprob.py")
FIXED = SOURCE.read_bytes()
ENV = os.environ.copy()
ENV["PYTHONDONTWRITEBYTECODE"] = "1"
ENV["TOKENIZERS_PARALLELISM"] = "false"

def execute(name, targets):
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-ra", *targets, "--junitxml=" + name + ".xml"]
    with Path(name + ".log").open("w") as output:
        result = subprocess.run(command, env=ENV, stdout=output, stderr=subprocess.STDOUT)
    print(Path(name + ".log").read_text(), flush=True)
    summary = {"returncode": result.returncode}
    if Path(name + ".xml").exists():
        suites = ET.parse(name + ".xml").getroot().iter("testsuite")
        for key in ("tests", "failures", "errors", "skipped"):
            summary[key] = sum(int(s.get(key, "0")) for s in ET.parse(name + ".xml").getroot().iter("testsuite"))
        summary["passed"] = summary["tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
    print(name, json.dumps(summary), flush=True)
    return summary

def main():
    target = ["tests/v1/worker/cpu/test_cpu_logprob.py", "-k", "preserve_common_offsets or uniform_large_offset or preserve_masked_and_nonfinite or empty_selected_tokens or topk_and_prompt_scores_with_common_offset or standalone_cpu_engine_logprobs"]
    try:
        SOURCE.write_bytes(subprocess.check_output(["git", "show", BASE + ":" + str(SOURCE)]))
        before = execute("cpu_logprob_before", target)
    finally:
        SOURCE.write_bytes(FIXED)
    after = execute("cpu_logprob_after", target)
    existing = execute("cpu_logprob_existing", [
        "tests/v1/worker/cpu/test_cpu_logprob.py", "-k",
        "not (preserve_common_offsets or uniform_large_offset or preserve_masked_and_nonfinite or empty_selected_tokens or topk_and_prompt_scores_with_common_offset or standalone_cpu_engine_logprobs)",
    ])
    evidence = {"base": BASE, "source_sha": "8b376970f56d33e8a543ee90fa69978cffe0d5d0", "before": before, "after": after, "existing": existing}
    Path("cpu_logprob_validation_result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert before.get("failures") > 0 and before.get("tests") == 28 and before.get("errors") == 0, before
    assert after.get("passed") == 28 and after.get("returncode") == 0 and after.get("skipped") == 0, after
    assert existing["returncode"] == 0, existing

if __name__ == "__main__":
    main()
