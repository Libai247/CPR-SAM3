"""Build a frozen DINOv3-LVD bank from unlabeled RGB images only."""

import argparse
import hashlib
import json
from pathlib import Path

from _common import add_runtime_arguments, setup, write_run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--images", required=True, help="Unlabeled reference RGB directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--seed-tokens", type=int, default=96)
    parser.add_argument("--image-list", help="Optional UTF-8 paths relative to --images, one per line")
    args = parser.parse_args()
    if min(args.max_images, args.clusters, args.seed_tokens) <= 0:
        parser.error("Image, cluster and token limits must be positive")
    root = Path(args.images).resolve()
    if not root.is_dir():
        parser.error("--images must be a directory")
    paths, profile = setup(args)
    import torch
    from PIL import Image
    from mmseg.registry import MODELS
    from cpr_sam3.runtime import model_config
    from cpr_sam3.prototypes import image_paths, image_prototypes, clustered_prototypes

    _, spec = model_config(profile, paths)
    model = MODELS.build(spec).eval().requires_grad_(False)
    selected = image_paths(root, args.max_images)
    if args.image_list:
        selected = []
        for line in Path(args.image_list).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path = (root / line.strip()).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                parser.error(f"Image-list entry is missing or outside --images: {line}")
            selected.append(path)
        if len(set(selected)) != len(selected):
            parser.error("Image list contains duplicates")
        if len(selected) > args.max_images:
            parser.error("Image list exceeds --max-images; supply the exact reference list")
    if not selected:
        parser.error("No reference images found")
    records, reference_manifest, dimension = [], [], None
    torch.manual_seed(0)
    with torch.inference_mode():
        for index, path in enumerate(selected, 1):
            image = Image.open(path).convert("RGB")
            logits, _, features = model.calibration_observation(image)
            dimension = features.shape[0]
            records.extend(image_prototypes(logits, features, args.seed_tokens))
            reference_manifest.append({
                "image": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            print(f"[{index}/{len(selected)}] {path.name}", flush=True)
    prototypes, valid, counts = clustered_prototypes(records, model.num_cls, args.clusters, dimension)
    if not valid.any():
        raise RuntimeError("No valid class prototypes were produced; inspect prompts and reference images")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        prototypes=prototypes, valid=valid, counts=counts,
        uses_ground_truth=False, domain="lvd", max_side=512,
        class_names=[entry["name"] for entry in json.loads(Path(spec["prompt_bank_path"]).read_text())["classes"]],
    ), output)
    manifest = dict(
        dataset=args.dataset, sampling="explicit_list" if args.image_list else "sorted_uniform_linspace",
        references=reference_manifest, uses_ground_truth=False,
        class_record_counts=counts.tolist(), clusters=args.clusters, seed_tokens=args.seed_tokens,
        domain="lvd", max_side=512,
        transductive_if_references_overlap_evaluation=True,
    )
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_record(output.parent / (output.stem + "_build"), args, paths)
    print(f"Saved {output}; class records: {counts.tolist()}")


if __name__ == "__main__":
    main()
