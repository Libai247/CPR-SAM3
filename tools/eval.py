"""Evaluate baseline or fixed CPR-SAM3 predictions with MMSegmentation."""

import argparse
from pathlib import Path

from _common import add_runtime_arguments, setup, write_run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--data-root", required=True, help="Prepared benchmark root, not its parent")
    parser.add_argument("--bank", help="Prototype .pt bank (required unless --baseline)")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if not args.baseline and (not args.bank or not Path(args.bank).is_file()):
        parser.error("--bank must name an existing prototype bank for CPR-SAM3")
    paths, profile = setup(args)
    from cpr_sam3.runtime import model_config
    from mmengine.runner import Runner

    cfg, model = model_config(profile, paths, args.bank, args.baseline)
    cfg.model = model
    cfg.data_root = str(Path(args.data_root).resolve())
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.num_workers = args.workers
    cfg.test_dataloader.persistent_workers = args.workers > 0
    cfg.launcher = "none"
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.randomness = dict(seed=0, deterministic=True)
    cfg.test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mFscore"])
    write_run_record(cfg.work_dir, args, paths)
    runner = Runner.from_cfg(cfg)
    result = runner.test()
    import json
    (Path(cfg.work_dir) / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
