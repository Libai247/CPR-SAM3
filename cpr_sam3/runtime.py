"""Resolve local assets without embedding the authors' machine paths."""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimePaths:
    upstream: Path
    dino_repo: Path
    sam3_checkpoint: Path
    dino_checkpoint: Path
    sam3_bpe: Path

    @classmethod
    def load(cls, filename):
        source = Path(filename).expanduser().resolve()
        values = json.loads(source.read_text(encoding="utf-8"))

        def resolve(name):
            path = Path(os.path.expandvars(values[name])).expanduser()
            return (source.parent / path).resolve() if not path.is_absolute() else path.resolve()

        return cls(*(resolve(key) for key in (
            "segearthov3_repo", "dinov3_repo", "sam3_checkpoint",
            "dinov3_checkpoint", "sam3_bpe",
        )))

    def problems(self, require_dino=True):
        checks = [
            (self.upstream / "segearthov3_segmentor.py", "file"),
            (self.upstream / "custom_datasets.py", "file"),
            (self.sam3_checkpoint, "file"),
            (self.sam3_bpe, "file"),
        ]
        if require_dino:
            checks.extend(((self.dino_repo / "hubconf.py", "file"), (self.dino_checkpoint, "file")))
        return [f"Missing {kind}: {path}" for path, kind in checks if not path.is_file()]


def normalized_hash(path):
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initialize(paths, allow_upstream_drift=False, require_dino=True):
    """Import the external baseline and override only its local asset paths."""
    problems = paths.problems(require_dino=require_dino)
    if problems:
        raise FileNotFoundError("\n".join(problems))
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    expected = provenance["external_baseline"]["normalized_sha256"]
    actual = normalized_hash(paths.upstream / "segearthov3_segmentor.py")
    if actual != expected and not allow_upstream_drift:
        raise RuntimeError(
            "The external SegEarth-OV3 baseline differs from the source used to assemble "
            "this release. See docs/REPRODUCIBILITY.md. Use --allow-upstream-drift only "
            "for a deliberately changed experiment, not an exact reproduction."
        )
    sys.path.insert(0, str(paths.upstream))
    os.environ["DINOV3_REPO"] = str(paths.dino_repo)
    baseline = importlib.import_module("segearthov3_segmentor")
    if Path(baseline.__file__).resolve() != paths.upstream / "segearthov3_segmentor.py":
        raise RuntimeError("A different segearthov3_segmentor is already imported.")
    if not getattr(baseline.build_sam3_image_model, "_cpr_asset_adapter", False):
        original = baseline.build_sam3_image_model

        @functools.wraps(original)
        def build_with_assets(*args, **kwargs):
            kwargs["bpe_path"] = str(paths.sam3_bpe)
            kwargs["checkpoint_path"] = str(paths.sam3_checkpoint)
            return original(*args, **kwargs)

        build_with_assets._cpr_asset_adapter = True
        baseline.build_sam3_image_model = build_with_assets
    importlib.import_module("custom_datasets")
    from mmengine.registry import init_default_scope

    init_default_scope("mmseg")


def load_profile(name):
    if name not in {"vaihingen", "potsdam", "loveda", "openearthmap"}:
        raise ValueError(f"Unsupported benchmark: {name}")
    return json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))


def model_config(profile, paths, bank=None, baseline=False):
    from mmengine.config import Config

    config = Config.fromfile(str(paths.upstream / profile["upstream_config"]))
    model = dict(config.model)
    model["classname_path"] = str(paths.upstream / model["classname_path"])
    if baseline:
        model["type"] = "SegEarthOV3Segmentation"
        return config, model
    from cpr_sam3.model import CPRSAM3Segmentor  # noqa: F401

    model.update(
        type="CPRSAM3Segmentor",
        prompt_bank_path=str(ROOT / "configs" / "prompt_banks" / profile["prompt_bank"]),
        prototype_bank_path=str(Path(bank).resolve()) if bank else None,
        dino_weights=str(paths.dino_checkpoint),
        dino_domain="lvd",
        dino_max_side=512,
        output_family=profile["output_family"],
    )
    return config, model


def environment_record(paths):
    import importlib.metadata
    import platform

    packages = {}
    for package in ("torch", "torchvision", "mmcv", "mmengine", "mmsegmentation", "numpy", "Pillow", "sam3"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not available as distribution metadata"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "baseline_normalized_sha256": normalized_hash(paths.upstream / "segearthov3_segmentor.py"),
    }
