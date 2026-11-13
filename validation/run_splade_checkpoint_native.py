import subprocess
import sys
from pathlib import Path

command = [sys.executable, "-B", "-m", "pytest", "-v", "-s", "tests/models/language/pooling/test_splade_checkpoint_loading.py::test_splade_checkpoint_aliases[canonical_bias-False]", "--junitxml=splade_after.xml"]
with Path("splade_after.log").open("w") as output:
    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
print(Path("splade_after.log").read_text(), flush=True)
raise SystemExit(result.returncode)
