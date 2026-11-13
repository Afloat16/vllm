import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = "191e5d5df2ec693d5536a635079d58864dfacebb"
SOURCE = Path("vllm/model_executor/models/bert.py")
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
    target = ["tests/models/language/pooling/test_head_dtype.py", "-k", "mlm_head_dtype"]
    after = execute("mlm_dtype_after", target)
    existing = execute("mlm_dtype_existing", [
        "tests/model_executor/layers/test_pooler_heads.py",
        "tests/model_executor/layers/test_pooler_activations.py",
    ])
    evidence = {"base": BASE, "source_sha": "a3af37f3e544a4434a52f75414731a10ae176303", "baseline_run": 43424033592, "after": after, "existing": existing}
    Path("mlm_dtype_validation_result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert after.get("passed") == 14 and after.get("returncode") == 0 and after.get("skipped") == 0, after
    assert existing["returncode"] == 0, existing

if __name__ == "__main__":
    main()
