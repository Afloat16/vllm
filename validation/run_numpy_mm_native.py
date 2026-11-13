import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = "191e5d5df2ec693d5536a635079d58864dfacebb"
SOURCES = [Path(name) for name in ("vllm/utils/serial_utils.py", "vllm/multimodal/media/image.py", "vllm/multimodal/media/video.py")]
FIXED = {path: path.read_bytes() for path in SOURCES}
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
    target = ["tests/renderers/test_numpy_embedding_byteorder.py"]
    try:
        for path in SOURCES:
            path.write_bytes(subprocess.check_output(["git", "show", BASE + ":" + str(path)]))
        before = execute("numpy_mm_before", target)
    finally:
        for path, content in FIXED.items():
            path.write_bytes(content)
    after = execute("numpy_mm_after", target)
    existing = execute("numpy_mm_existing", [
        "tests/utils_/test_serial_utils.py",
        "tests/renderers/test_sparse_tensor_validation.py",
        "-k", "numpy or encode_and_decode",
    ])
    evidence = {"base": BASE, "source_sha": "c736616c6a56a2d9f073fe6df952f179a114376e", "before": before, "after": after, "existing": existing}
    Path("numpy_mm_validation_result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert before.get("failures") == 27 and before.get("passed") == 23 and before.get("errors") == 0, before
    assert after.get("passed") == 50 and after.get("returncode") == 0 and after.get("skipped") == 0, after
    assert existing["returncode"] == 0, existing

if __name__ == "__main__":
    main()
