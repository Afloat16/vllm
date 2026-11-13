import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "c74cab9b33659e92de9ca9c35eb949c78ebbe64c"
HEAD = "d6bb1261e68b20fe8eb4c1a74860c17d3f6e452b"
FILES = ["vllm/entrypoints/generate/beam_search/offline.py", "tests/samplers/test_beam_search_batch.py"]
SOURCE_FILES = [path for path in FILES if path.startswith("vllm/entrypoints/")]
TEST = "tests/samplers/test_beam_search_batch.py"
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
native_changes = subprocess.check_output(
    ["git", "diff", "--name-only", BASE, HEAD, "--", "csrc", "cmake", "rust", "CMakeLists.txt", "setup.py"],
    text=True,
)
assert not native_changes, native_changes
import torch
import vllm
native = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in Path("vllm").glob("*.so")}
assert native, "Source-built CPU extensions are missing"
print("Native runtime", torch.__version__, vllm.__version__, native, flush=True)

def run(label, path, expression=None):
    args = [sys.executable, "-m", "pytest", path, "-q", f"--junitxml={label}.xml"]
    if expression:
        args += ["-k", expression]
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    root = ElementTree.parse(f"{label}.xml").getroot()
    totals = {key: sum(int(node.get(key, 0)) for node in root.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    print(label, totals, flush=True)
    return totals

fixed = {path: Path(path).read_bytes() for path in SOURCE_FILES}
try:
    for path in SOURCE_FILES:
        Path(path).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{path}"]))
    before = run("before", TEST, None)
finally:
    for path, content in fixed.items():
        Path(path).write_bytes(content)
assert before["tests"] == 26 and before["errors"] == before["skipped"] == 0, before
assert before["returncode"] == 1 and before["failures"] > 0, before
after = run("after", TEST, None)
assert after == {"tests": 26, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, after
existing = run("existing", "tests/samplers/test_beam_search_online.py")
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in FILES:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
result = {"head": HEAD, "base": BASE, "runtime_build_run": 43302132651,
          "native_extensions": native, "torch": torch.__version__, "vllm": vllm.__version__,
          "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
