# Local Validation

Source-release preparation date: 2026-10-26.

Performed in a CPU-only Python 3.12.14 / PyTorch 2.14.1+cpu runtime:

- Nine unit tests passed with `python -m unittest discover -s tests -v`.
- Prototype pooling/clustering, trajectory gates and evidence ties were checked
  with synthetic tensors using real CPU PyTorch.
- The public model's import closure and final-output selection were checked
  using lightweight framework stubs. This is not an actual MMseg model build.
- Five command-line `--help` entry points passed without model dependencies.
- Source provenance, Python syntax, local import closure, Markdown links,
  prompt banks, private paths and credential-pattern checks passed.

CUDA was unavailable. No SAM3 or DINOv3 checkpoint was loaded. No dataset was
evaluated, no real inference image was generated, and no fresh GPU environment
installation was validated. The reference metric table is not a test result.

The GitHub workflow runs CPU tests and source checks only. Passing that workflow
does not complete the benchmark checklist in REPRODUCIBILITY.md.
