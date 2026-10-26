"""Read-only dependency and local asset checks; never download weights."""

import argparse
import importlib.util
import json
import sys

from _common import ROOT
from cpr_sam3.runtime import RuntimePaths, normalized_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="paths.json")
    args = parser.parse_args()
    problems = []
    for name in ("torch", "mmengine", "mmcv", "mmseg", "sam3", "numpy", "PIL"):
        present = importlib.util.find_spec(name) is not None
        print(f"{'OK' if present else 'MISSING'} package {name}")
        if not present:
            problems.append(f"Missing package: {name}")
    try:
        paths = RuntimePaths.load(args.paths)
        problems.extend(paths.problems())
        baseline = paths.upstream / "segearthov3_segmentor.py"
        if baseline.is_file():
            expected = json.loads((ROOT / "provenance.json").read_text())["external_baseline"]["normalized_sha256"]
            actual = normalized_hash(baseline)
            print(f"Baseline SHA256: {actual}")
            if actual != expected:
                problems.append("Baseline source hash differs; see docs/REPRODUCIBILITY.md")
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"Paths configuration: {error}")
    if importlib.util.find_spec("torch") is not None:
        import torch
        print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            problems.append("CUDA is required for SAM3 inference")
    for problem in problems:
        print(f"ERROR: {problem}")
    print("Ready for GPU smoke testing" if not problems else "Preflight incomplete")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
