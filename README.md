# CPR-SAM3

**Complementary Prediction Reconciliation for Training-Free Remote-Sensing
Open-Vocabulary Semantic Segmentation**

Huasong Chen, Wenting Li, Chuihao Meng, Tianzuo Nie, Yongjun Zhang

Research code for recovering alternative SAM3 predictions, evaluating their
appearance with frozen DINOv3 prototypes, and reconciling candidate conflicts.
Neither backbone is trained by this repository.

## Release Status

This is a source-release draft assembled from the existing research code.
CPU helper tests and source checks are provided. End-to-end CUDA inference,
fresh-machine installation and exact reproduction of the manuscript tables
have **not** been validated for this packaged release. See
[reproducibility notes](docs/REPRODUCIBILITY.md) before interpreting results.
The preserved implementation contains additional gates and routing rules
beyond the manuscript's compact mathematical description.

## Repository Layout

```text
cpr_sam3/
  model.py             Fixed deployable result and prototype-bank validation
  runtime.py           External dependencies and local paths
  prototypes.py        Original image pooling and clustering helpers
  research/            Preserved method implementation and its import closure
configs/
  *.json               Four benchmark profiles
  prompt_banks/        Class prompts from the research implementation
tools/
  doctor.py            Check assets and dependencies without inference
  build_prototypes.py  Build a bank from unlabeled RGB images
  demo.py              Predict one image without annotations
  eval.py              Evaluate baseline or CPR-SAM3
  check_release.py     Check source hashes, imports and accidental private files
tests/                 CPU-only unit tests
docs/                  Data, reproducibility and third-party notes
```

## Dependencies and Installation

Start from a working **SegEarth-OV3 + SAM3** CUDA environment. Do not reuse
SCLIP's older PyTorch installation commands for SAM3. Follow the official
[SAM3 installation instructions](https://github.com/facebookresearch/sam3)
and the matching PyTorch/CUDA installation instructions for your machine.
MMSegmentation's MMCV build must also match that environment. This repository
does not claim that a new environment has been tested across all CUDA versions.

Obtain the external source trees yourself; they are not redistributed here:

```bash
git clone https://github.com/earth-insights/SegEarth-OV-3.git third_party/SegEarth-OV-3
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
python -m pip install -r requirements.txt
```

Install SAM3 according to its official instructions. This code uses the
`Sam3Processor` image API and accesses decoder tensors; later SAM3 API changes
can break compatibility. Preserve the versions from your successful experiment
environment. The baseline's content hash is checked at startup; details are in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Checkpoints and Paths

Obtain the original SAM3 checkpoint through its official distribution, and
the **DINOv3 ViT-L/16 LVD-1689M** checkpoint through the
[official DINOv3 repository](https://github.com/facebookresearch/dinov3).
Accept any required terms yourself. SAT-493M weights are not interchangeable
with the LVD configuration in this release. No weights or tokens are bundled
or automatically downloaded.

Copy `paths.example.json` to `paths.json` and edit its five paths. Relative
paths are resolved relative to that JSON file, not the shell's working directory.
`sam3_bpe` must point to the actual `bpe_simple_vocab_16e6.txt.gz` asset in your
SAM3 installation or checkout; the example is only a conventional location.

```bash
python tools/doctor.py --paths paths.json
```

## Datasets

Prepare Vaihingen, Potsdam, LoveDA and OpenEarthMap using
[DATASETS.md](docs/DATASETS.md). Dataset images and labels remain external.
The profiles reuse the original upstream data layout, class mapping,
background handling and sliding-window configuration.

## Build a Prototype Bank

Bank construction reads only RGB reference images and SAM3 outputs, not labels:

```bash
python tools/build_prototypes.py --paths paths.json --dataset vaihingen --images data/Vaihingen/img_dir/val --output artifacts/vaihingen_lvd.pt
```

By default, at most 64 images are sampled uniformly from a sorted image list;
up to three prototypes per class and 96 seed tokens per image/class are used.
The sidecar JSON records the selected relative image names, image hashes and
sampling settings. To reuse a known reference split, pass
`--image-list splits/vaihingen_reference.txt`, with one relative RGB filename
per line. Do not silently substitute a different reference set and call the
result an exact reproduction.

Using reference images from the evaluation partition is **unlabeled target-domain
adaptation / transductive inference**, not an image-disjoint evaluation. For an
image-disjoint experiment, use reference images from a separate source and
verify the split before evaluating. See the protocol notes for limitations.

## Single-Image Inference

```bash
python tools/demo.py --paths paths.json --dataset vaihingen --image data/Vaihingen/img_dir/val/IMAGE.png --bank artifacts/vaihingen_lvd.pt --output-dir work_dirs/demo
```

Replace `IMAGE.png` with a real prepared image. Outputs are an indexed label
PNG, a color PNG and a run-environment record. No annotation path is required.
Add `--baseline` and omit `--bank` to inspect the SegEarth-OV3 reference.

## Evaluation

```bash
python tools/eval.py --paths paths.json --dataset vaihingen --data-root data/Vaihingen --baseline --work-dir work_dirs/vaihingen_baseline
python tools/eval.py --paths paths.json --dataset vaihingen --data-root data/Vaihingen --bank artifacts/vaihingen_lvd.pt --work-dir work_dirs/vaihingen_cpr
```

The other profiles use the same commands:

| `--dataset` | `--data-root` example | Bank example |
| --- | --- | --- |
| `potsdam` | `data/Potsdam` | `artifacts/potsdam_lvd.pt` |
| `loveda` | `data/LoveDA` | `artifacts/loveda_lvd.pt` |
| `openearthmap` | `data/OpenEarthMap` | `artifacts/openearthmap_lvd.pt` |

Create each bank using that profile and its corresponding RGB reference folder.
Run from this repository root. `metrics.json`, MMEngine logs and `run.json`
are saved under `--work-dir`. The default worker count is zero for portability;
use `--workers 4` only after a successful smoke run. This release exposes one
fixed output per profile; it never chooses the best result after reading IoU.

## Reproduction Checks

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

These are CPU/source checks, not evidence of CUDA or benchmark parity.
Manuscript reference numbers are in
[reference_metrics.json](results/reference_metrics.json), clearly marked as
reported values rather than outputs generated by this release.

## Citation and Acknowledgements

The manuscript is not marked as accepted or published here. Citation metadata
is provided in [CITATION.cff](CITATION.cff).

This work builds on
[SegEarth-OV3](https://github.com/earth-insights/SegEarth-OV-3),
[SAM3](https://github.com/facebookresearch/sam3),
[DINOv3](https://github.com/facebookresearch/dinov3) and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
Related release layouts were inspected in
[SegEarth-OV](https://github.com/likyoo/SegEarth-OV),
[SCLIP](https://github.com/wangf3014/SCLIP),
[ProxyCLIP](https://github.com/mc-lan/ProxyCLIP) and
[CorrCLIP](https://github.com/zdk258/CorrCLIP); their README text is not copied.
See [third-party notes](docs/THIRD_PARTY.md).

No project-wide `LICENSE` is added to this draft. Public source availability
alone does not grant unrestricted reuse rights; the authors can add an explicit
license after confirming rights. Third-party dependencies and checkpoints
retain their own applicable terms.
