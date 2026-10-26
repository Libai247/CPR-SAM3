<div align="center">

<h1>CPR-SAM3</h1>
<h3>Complementary Prediction Reconciliation for Training-Free<br>Remote-Sensing Open-Vocabulary Semantic Segmentation</h3>

<p>
Huasong Chen<sup>1</sup>, Wenting Li<sup>2</sup>, Chuihao Meng<sup>1</sup>,
Tianzuo Nie<sup>1</sup>, Yongjun Zhang<sup>1</sup>
</p>
<p><sup>1</sup>Guizhou University &nbsp; <sup>2</sup>Guizhou University of Commerce</p>

[Overview](#overview) | [Results](#results) | [Installation](#installation) | [Usage](#usage) | [Citation](#citation)

</div>

## Overview

CPR-SAM3 is a training-free framework for remote-sensing open-vocabulary
semantic segmentation. It recovers alternative labels from SAM3 outputs and
uses frozen DINOv3 features to reconcile conflicting predictions. Both
foundation models remain frozen; no ground-truth masks are used to build the
prototype bank or predict an image.

[![CPR-SAM3 framework: candidate recovery, visual verification, and prediction reconciliation](assets/framework.png)](assets/framework.png)

- **Complementary candidate recovery (CCR):** collect hypotheses from SAM3's semantic head, instance head, decoder layers, and an aligned view.
- **Cross-foundation visual verification (CFVV):** assess candidate labels using DINOv3 class prototypes and neighborhood agreement.
- **Reliability-constrained reconciliation (RCR):** combine cross-output support with visual evidence to obtain the final segmentation.

## Results

Semantic segmentation results (mIoU, %):

| Method | Vaihingen | Potsdam | LoveDA | OpenEarthMap | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SegEarth-OV3 | 60.76 | 57.85 | 47.37 | 44.10 | 52.52 |
| **CPR-SAM3** | **62.93** | **60.15** | **49.59** | **47.23** | **54.98** |

These are reported manuscript results, not fresh measurements from this
release. See [reproducibility notes](docs/REPRODUCIBILITY.md) for the evaluation
protocol and remaining validation work.

<details>
<summary>Qualitative comparisons on OpenEarthMap</summary>

![OpenEarthMap comparisons from the manuscript](assets/qualitative.png)

Examples from the manuscript, comparing SCLIP, ClearCLIP, ResCLIP,
SegEarth-OV3, and CPR-SAM3. These are not outputs of a new release-validation run.

</details>

## Installation

A CUDA GPU and a working **SAM3 + MMSegmentation** environment are required.
Install [SAM3](https://github.com/facebookresearch/sam3) and a compatible
PyTorch / MMCV / MMSegmentation stack following their official instructions.
Then clone this repository and its external dependencies:

```bash
git clone https://github.com/Libai247/CPR-SAM3.git
cd CPR-SAM3
git clone https://github.com/earth-insights/SegEarth-OV-3.git third_party/SegEarth-OV-3
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
python -m pip install -r requirements.txt
```

Download the **SAM3** and **DINOv3 ViT-L/16 LVD-1689M** checkpoints from their
official repositories. Copy [paths.example.json](paths.example.json) to
`paths.json` and set the two source directories, two checkpoint paths, and SAM3
tokenizer path. Paths are relative to the JSON file. Weights are not bundled.

```bash
cp paths.example.json paths.json
# Edit paths.json to match your local files.
python tools/doctor.py --paths paths.json
```

The dependency list is not a tested CUDA environment lockfile. See
[validation status](docs/VALIDATION.md) and [compatibility notes](docs/REPRODUCIBILITY.md).

## Usage

Prepare the datasets using [Data Preparation](docs/DATASETS.md). Run the
following commands from the repository root.

### 1. Build the Prototype Bank

Build a bank from unlabeled reference images for the selected dataset:

```bash
python build_prototypes.py --dataset vaihingen \
  --images data/Vaihingen/img_dir/val \
  --output artifacts/vaihingen_lvd.pt
```

The default uses at most 64 reference images and up to three prototypes per
class. A JSON sidecar records the selected images and settings. Using images
from the evaluation partition is a **transductive** setting; use a separate
reference split for image-disjoint evaluation.

### 2. Single-Image Inference

```bash
python demo.py --dataset vaihingen \
  --image data/Vaihingen/img_dir/val/IMAGE.png \
  --bank artifacts/vaihingen_lvd.pt --output-dir work_dirs/demo
```

Replace `IMAGE.png` with an existing image. The output directory contains
`prediction.png` (class IDs), `prediction_rgb.png`, and `run.json`.

### 3. Evaluation

```bash
python eval.py --dataset vaihingen --data-root data/Vaihingen \
  --bank artifacts/vaihingen_lvd.pt --work-dir work_dirs/vaihingen
```

To evaluate the SegEarth-OV3 baseline, replace `--bank ...` with `--baseline`.
Results and logs are saved under `--work-dir`. Use the same data and class
mapping for both methods.

| `--dataset` | Prepared dataset root | RGB reference directory |
| :--- | :--- | :--- |
| `vaihingen` | `data/Vaihingen` | `img_dir/val` |
| `potsdam` | `data/Potsdam` | `img_dir/val` |
| `loveda` | `data/LoveDA` | `img_dir/val` |
| `openearthmap` | `data/OpenEarthMap` | `val/images` |

Build a separate bank for each dataset. The profiles and text prompts are in
[configs](configs). All commands accept `--paths` and `--help`.

## Citation

```bibtex
@misc{chen2026cprsam3,
  title  = {CPR-SAM3: Complementary Prediction Reconciliation for Training-Free Remote-Sensing Open-Vocabulary Semantic Segmentation},
  author = {Chen, Huasong and Li, Wenting and Meng, Chuihao and Nie, Tianzuo and Zhang, Yongjun},
  year   = {2026},
  note   = {Unpublished manuscript},
  url    = {https://github.com/Libai247/CPR-SAM3}
}
```

## Acknowledgements

This implementation builds on [SegEarth-OV3](https://github.com/earth-insights/SegEarth-OV-3),
[SAM3](https://github.com/facebookresearch/sam3),
[DINOv3](https://github.com/facebookresearch/dinov3), and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
We thank their authors for sharing their work. See [third-party notes](docs/THIRD_PARTY.md)
for source and asset information.
