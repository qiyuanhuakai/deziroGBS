"""Run the four frozen configurations sequentially with a single cache writer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = args.output_root.resolve()
environment = dict(os.environ)
environment["PYTHONPATH"] = str(root / "src")
environment["OPENBLAS_NUM_THREADS"] = "1"
environment["OMP_NUM_THREADS"] = "1"
for config_name, module in [("survey", "experiment"), ("adaptive", "experiment"),
                             ("heldout", "experiment"), ("scaling", "scaling")]:
    command = [sys.executable, "-m", f"gbslab.{module}", "--config",
               str(root / "configs" / f"{config_name}.json"),
               "--output", str(output / config_name)]
    print(f"Running {config_name}", flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
