# Third-Party Sources and Assets

## External Code (Not Vendored)

- SegEarth-OV3: https://github.com/earth-insights/SegEarth-OV-3
  supplies the SAM3 baseline class, class-name files, dataset registration and
  benchmark base configs. Obtain it from the original authors. No baseline
  source files or Git history are copied into this release.
- SAM3: https://github.com/facebookresearch/sam3
  supplies the promptable segmentation model and its tokenizer asset.
- DINOv3: https://github.com/facebookresearch/dinov3
  supplies the frozen dense visual encoder.
- MMSegmentation / MMEngine / MMCV:
  https://github.com/open-mmlab/mmsegmentation
  and their original distributions supply framework components.

Each dependency retains its own license and asset terms. This repository does
not relicense or redistribute the external source trees.

## Related Research

SegEarth-OV3 builds on SAM3 and SCLIP. See its original repository for the
upstream acknowledgements and references.

## Weights and Data

No model checkpoint, tokenizer asset, standalone dataset sample or access
token is included. Download weights and data from the original providers under
their applicable terms. Keep local paths, credentials, and restricted assets
out of public commits.

## Manuscript Figures

`assets/framework.png` and `assets/qualitative.png` are the authors' existing
manuscript Figures 3 and 5, respectively, copied without changing the depicted
method or predictions. The figures contain benchmark image crops; the
underlying imagery remains subject to the original dataset terms:
[ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx)
and [OpenEarthMap](https://www.open-earth-map.org/).
