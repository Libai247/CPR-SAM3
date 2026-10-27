# Implementation and Evaluation Notes

## What Is Preserved

The implementation of `CrossExpertTrajectorySegmentor` and its dependencies
is retained under `cpr_sam3/research/`. Historical module names are kept for
comparison with the experiment code. [provenance.json](../provenance.json)
records the source files and hashes. Changes to those files are limited to
imports and default asset paths.

The public evaluator measures the configured final prediction with standard
IoU metrics. Historical diagnostic metrics are not imported, and ground-truth
oracle comparisons do not select the output.

## Upstream Version

The installation command pins SegEarth-OV3 to
[`87e5cc18c8bc9c0b4713b975587281ac1c97c79a`](https://github.com/earth-insights/SegEarth-OV-3/tree/87e5cc18c8bc9c0b4713b975587281ac1c97c79a).
Its baseline file matches the normalized SHA256 in `provenance.json`, checked
on 2026-10-27. The baseline is loaded from that external checkout rather than
copied into this repository.

Startup checks the baseline hash. To test a different version, pass
`--allow-upstream-drift` and record that version with the results. The pinned
baseline does not pin the SAM3 or DINOv3 packages, nor recover the original
experiment environment.

## Fixed Outputs

Each benchmark profile specifies one output family:

| Benchmark profile | Output family |
| --- | --- |
| Vaihingen | `direct_vote` |
| Potsdam | `presence_vote` |
| LoveDA | `schema_route` |
| OpenEarthMap | `direct_raw` |

These settings come from the research visualization configs. The runner does
not read labels or IoUs to choose a family. They are benchmark-specific
settings, not a validated universal routing policy; selecting a different
family after checking test mIoU would be test-label tuning.

## Differences from the Manuscript Description

The manuscript equations describe the overall procedure, but are not a
complete specification of the implementation. The code also includes:

- Fixed response and class-schema hypotheses.
- Candidate confirmation/veto and prototype rescue gates.
- A trajectory score containing a vote term, a visual-improvement condition,
  a local-support threshold and a bounded change budget.
- Evidence-max reconciliation between the response and rotation branches.

Setting the minimum vote count to one leaves the other gates active. In a
candidate-union-only formulation, that vote threshold alone cannot exclude
members of the union. This distinction matters for interpreting module
ablations or implementing the compact formula independently.

Exact agreement between this packaged implementation, the manuscript equations,
and the ablation definitions has not been established. It still requires the
original checkpoint versions, reference-image lists, and raw evaluation logs.

## Prototype Protocol

The builder uses the original sorted-uniform image sampling, quality-weighted
pooling, and clustering functions with frozen DINOv3 ViT-L/16 and LVD
normalization. It records the reference RGB images and their hashes and reads
no annotation masks. The original banks are not bundled. Reference images,
preprocessing, encoder revisions, and checkpoints can all affect a rebuilt bank.

The README example is transductive because its reference images come from the
evaluation partition. Image-disjoint evaluation needs a separate reference
source and an overlap check. Reference images, thresholds, and prompts must
be selected without test-label tuning.

## Environment and Performance

Each run records installed package versions. The original server environment
lockfile is unavailable; `requirements.txt` lists wrapper dependencies, not a
validated CUDA environment. SAM3 and MMCV compatibility still needs testing
in the intended GPU environment.

The research predictor computes multiple candidates before the wrapper selects
the configured output. Runtime and peak memory for this wrapper have not been
measured; manuscript efficiency figures cannot be assumed to apply. Prototype
construction is an additional cost and should be measured separately.

## Remaining GPU Checks

- [ ] Confirm upstream source and exact SAM3 / DINOv3 revisions.
- [ ] Run `tools/doctor.py` in the intended CUDA environment.
- [ ] Export the successful environment package list.
- [ ] Validate baseline on the original prepared image/label split.
- [ ] Check exact reference-image hashes or disclose a new bank.
- [ ] Run a one-image label-free demo and check class IDs and geometry.
- [ ] Run all four datasets and compare to the original raw logs.
- [ ] Audit manuscript equations and cumulative ablation definitions.
- [ ] Measure full inference and prototype construction separately.

CPU tests and source checks do not mark these GPU/benchmark items complete.
