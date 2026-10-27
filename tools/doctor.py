"""Read-only dependency and local asset checks; never download weights."""

import argparse
import importlib
import importlib.util
from importlib.machinery import PathFinder
import json
import sys

from _common import ROOT
from cpr_sam3.runtime import RuntimePaths, normalized_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="paths.json")
    args = parser.parse_args()
    problems = []
    paths = None
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

    available = {}
    for name in (
        "torch", "torchvision", "mmengine", "mmcv", "mmseg", "numpy", "PIL", "cv2",
        "timm", "ftfy", "regex", "einops", "iopath", "huggingface_hub", "tqdm",
        "scipy", "skimage", "pycocotools", "psutil", "hydra", "omegaconf",
        "typing_extensions", "packaging",
    ):
        available[name] = importlib.util.find_spec(name) is not None
        print(f"{'OK' if available[name] else 'MISSING'} package {name}")
        if not available[name]:
            problems.append(f"Missing package: {name}")

    bundled_sam3 = paths is not None and PathFinder.find_spec("sam3", [str(paths.upstream)]) is not None
    sam3_present = bundled_sam3 or importlib.util.find_spec("sam3") is not None
    source = " (SegEarth-OV3 checkout)" if bundled_sam3 else ""
    print(f"{'OK' if sam3_present else 'MISSING'} package sam3{source}")
    if not sam3_present:
        problems.append("Missing SAM3 source in the configured checkout or Python environment")

    for module in ("mmcv._ext", "mmseg.models"):
        if not available[module.split(".")[0]]:
            continue
        try:
            importlib.import_module(module)
            print(f"OK import {module}")
        except Exception as error:
            problems.append(f"Cannot import {module}: {error}")

    if available["torch"]:
        try:
            import torch
            print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}")
            if not torch.cuda.is_available():
                problems.append("CUDA is required for SAM3 inference")
        except Exception as error:
            problems.append(f"Cannot import PyTorch: {error}")
    for problem in problems:
        print(f"ERROR: {problem}")
    print("Ready for GPU smoke testing" if not problems else "Preflight incomplete")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
