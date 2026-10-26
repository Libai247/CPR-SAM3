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

Consult each upstream source and asset's current terms. Public availability
does not automatically authorize relicensing or redistribution. No project-wide
license is imposed on these dependencies by this release.

## Related Research

The original baseline and associated pipelines acknowledge SCLIP and other
foundation-model work. Preserve those upstream acknowledgements when using
their code. The layout of this release was informed by the official SCLIP,
SegEarth-OV, SegEarth-OV3, ProxyCLIP and CorrCLIP repositories. Layout inspection
does not imply their authors endorse this implementation.

## Weights and Data

No model checkpoint, tokenizer asset, dataset, original-image sample or access
token is included. Users obtain each asset directly through its official
provider and accept the associated terms. Do not upload access tokens or
restricted files to GitHub, Git LFS or a public release attachment.
