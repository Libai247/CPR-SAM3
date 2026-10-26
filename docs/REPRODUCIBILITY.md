# Reproducibility and Scope

## What Is Preserved

The release preserves the transitive source dependencies of
`CrossExpertTrajectorySegmentor`. Historical module names remain under
`cpr_sam3/research/` to make comparison against original experiment code easier.
The mechanical provenance manifest records source/destination files and hashes.
Only imports and private default asset paths are rewritten in these files.

Package initializers intentionally do not import the old diagnostic metrics.
The public evaluator uses the fixed final prediction with standard IoU metrics;
ground-truth oracle comparisons are not used to select a deployed output.

The external baseline is not redistributed. Its normalized SHA256 at assembly
is recorded in `provenance.json`. The matching baseline was compared with the
official repository on 2026-10-26. A future Git checkout may differ. Startup
rejects mismatches unless `--allow-upstream-drift` is explicitly supplied.
Changing it creates a different experimental setting, not a verified historical
reproduction. A content hash does not pin all SAM3/DINOv3 dependencies.

## Fixed Outputs

The source research visualization configs prescribe these evidence-max families:

| Benchmark profile | Output family |
| --- | --- |
| Vaihingen | `direct_vote` |
| Potsdam | `presence_vote` |
| LoveDA | `schema_route` |
| OpenEarthMap | `direct_raw` |

These are historical, explicit benchmark configurations. The public runner
does not read labels or IoUs to choose a family. Do not describe this release
as a universal, newly validated image-adaptive routing policy. Changing families
after observing test mIoU is not a label-free evaluation protocol.

## Manuscript / Implementation Audit

The compact manuscript equation is not a complete executable specification of
the preserved research implementation. The implementation also includes:

- Fixed response and class-schema hypotheses.
- Candidate confirmation/veto and prototype rescue gates.
- A trajectory score containing a vote term, a visual-improvement condition,
  a local-support threshold and a bounded change budget.
- Evidence-max reconciliation between the response and rotation branches.

In particular, setting the minimum vote count to one does not mean all the
other gates are disabled. Conversely, in a candidate-union-only mathematical
description, a one-vote gate alone cannot exclude union members. Resolve this
description-versus-code issue before claiming that a simplified formula
reproduces the reported result or interpreting a module-only ablation.

This release does not fabricate missing experiments, alter reported scores,
or silently replace the implementation with the simplified equation. The
original-run checkpoint versions, prototype reference lists and result
provenance must be reconciled with the manuscript before a final verified tag.

## Prototype Protocol

The builder copies the original deterministic sorted-uniform image selection,
quality-weighted pooling and clustering helpers. It uses LVD normalization and
frozen DINOv3 ViT-L/16. It records every selected RGB reference image and hash.
It reads no annotation masks. However, a different reference set, preprocessing
layout, backbone revision or checkpoint can change the bank and results.
Original historical banks are not bundled, so rebuilding alone does not prove
bitwise identity with the manuscript's bank.

When reference and evaluation images overlap, report the transductive setting.
Image-disjoint evaluation requires an independently selected reference source
and an explicit overlap audit. Test-label tuning must not be used to choose
reference images, thresholds or prompts.

## Environment and Performance

The wrapper records installed package versions for each inference run. No
complete original server lockfile was recovered into this source-only release.
`requirements.txt` is therefore not presented as a validated CUDA lockfile.
SAM3 and MMCV compatibility must be verified in the intended environment.

The preserved research predictor computes multiple historical candidates.
The public wrapper selects one afterward. This favors source fidelity over
optimized inference; do not attribute manuscript latency/memory numbers to
this wrapper without remeasurement. Bank construction cost is additional.

## Validation Checklist

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
