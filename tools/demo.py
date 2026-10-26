"""Run one image without loading any ground-truth annotations."""

import argparse
from pathlib import Path

from _common import add_runtime_arguments, setup, write_run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bank", help="Prototype .pt bank (required unless --baseline)")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not Path(args.image).is_file():
        parser.error("--image must be an existing RGB image")
    if not args.baseline and (not args.bank or not Path(args.bank).is_file()):
        parser.error("--bank must name an existing prototype bank for CPR-SAM3")
    paths, profile = setup(args)
    import numpy as np
    import torch
    from mmseg.registry import DATASETS, MODELS
    from mmseg.structures import SegDataSample
    from PIL import Image
    from cpr_sam3.runtime import model_config

    cfg, model_spec = model_config(profile, paths, args.bank, args.baseline)
    model = MODELS.build(model_spec).eval().requires_grad_(False)
    image_path = Path(args.image).resolve()
    with Image.open(image_path) as source:
        width, height = source.size
    sample = SegDataSample(metainfo=dict(img_path=str(image_path), ori_shape=(height, width)))
    with torch.inference_mode():
        result = model.predict(None, [sample])[0]
    prediction = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction).save(output / "prediction.png")
    dataset = DATASETS.get(cfg.dataset_type)
    palette = np.asarray(dataset.METAINFO["palette"], dtype=np.uint8)
    Image.fromarray(palette[prediction]).save(output / "prediction_rgb.png")
    write_run_record(output, args, paths)
    print(f"Saved prediction.png and prediction_rgb.png in {output}")


if __name__ == "__main__":
    main()
