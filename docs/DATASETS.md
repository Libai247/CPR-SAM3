# Data Preparation

Download datasets from their original providers and follow their terms:

- [ISPRS 2D semantic labeling](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx)
- [LoveDA](https://github.com/Junjue-Wang/LoveDA)
- [OpenEarthMap](https://www.open-earth-map.org/)

Use the preprocessing instructions in
[SegEarth-OV](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md)
and the dataset configs in your external SegEarth-OV3 checkout. This release
does not invent a new crop split or redistribute benchmark imagery.

## Expected Prepared Layout

```text
data/
  Vaihingen/
    img_dir/val/       RGB crops
    ann_dir/val/       Class-index masks
  Potsdam/
    img_dir/val/
    ann_dir/val/
  LoveDA/
    img_dir/val/
    ann_dir/val/
  OpenEarthMap/
    val/images/
    val/labels/
```

Image and annotation suffixes must match the external MMSeg dataset classes.
In particular, the external OpenEarthMap class defaults to `.tif` images and
`.tif` label masks. Do not silently rename files without checking pair matching.
Labels must be class-index masks, not RGB palette images.

## Taxonomy and Ignore Labels

- Vaihingen/Potsdam: six configured classes, fallback class index 5.
- LoveDA: the original config sets `reduce_zero_label=True`; raw no-data zero
  becomes ignore label 255, and raw classes 1-7 become model classes 0-6.
- OpenEarthMap: the research config uses nine entries, including index-zero
  background, and `reduce_zero_label=False`. The default metric ignores 255,
  not automatically zero. This configured evaluation differs from simply
  averaging eight foreground classes. State the convention when reporting it.

Never change the evaluated class set to match a target score. For fairness,
use exactly the same prepared data, split and class/ignore mapping for baseline
and full-method evaluation.

The manuscript reports 398 / 2016 / 1669 / 384 evaluation images for Vaihingen /
Potsdam / LoveDA / OpenEarthMap. Counts are sanity checks, not sufficient proof
that two splits are identical. The original reference-image manifests and exact
crop preprocessing lineage must be checked before claiming manuscript parity.

## Reference Images

The prototype builder emits a sidecar JSON with selected RGB filenames and
SHA256 hashes. Keep that file with your experiment outputs. Do not publish
machine-specific absolute paths, private images, or data without redistribution
permission. A reference folder inside the evaluation partition must be disclosed
as transductive, even though no labels are read during bank construction.
