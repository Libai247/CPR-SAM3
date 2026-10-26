from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_runtime_arguments(parser):
    parser.add_argument("--paths", default="paths.json", help="Local asset paths JSON")
    parser.add_argument("--dataset", required=True, choices=("vaihingen", "potsdam", "loveda", "openearthmap"))
    parser.add_argument("--allow-upstream-drift", action="store_true")


def setup(args):
    from cpr_sam3.runtime import RuntimePaths, initialize, load_profile

    paths = RuntimePaths.load(args.paths)
    initialize(paths, args.allow_upstream_drift, require_dino=not getattr(args, "baseline", False))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 inference in this release requires a CUDA GPU.")
    return paths, load_profile(args.dataset)


def write_run_record(directory, args, paths):
    import json
    from cpr_sam3.runtime import environment_record

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = environment_record(paths)
    record["arguments"] = vars(args)
    (directory / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
