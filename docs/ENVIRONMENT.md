# Environment

## Recorded Experiment Setup

The project's environment restore record (2026-07-27) and evaluation logs
(2026-09-29) record the following Linux setup:

| Component | Recorded version |
| :--- | :--- |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| GPU in the evaluation logs | NVIDIA RTX A6000 |
| NumPy | 1.26.4 |
| MMCV | 2.1.0, full package with CPU extensions |
| MMEngine | 0.10.7 |
| MMSegmentation | 1.2.2 |
| OpenCV (`cv2.__version__`) | 4.11.0 |

The core pins in `requirements.txt` follow these records. `opencv-python`
matches MMEngine's published wheel dependency and the recorded OpenCV version;
the logs do not identify the original OpenCV wheel distribution. The `ftfy`
pin follows [SAM3's package metadata](https://github.com/facebookresearch/sam3/blob/main/pyproject.toml).
Other entries list import dependencies rather than recovered historical pins.
The complete original `pip freeze` and conda-pack archive have not been
recovered into this public release.

## PyTorch

Create a separate environment, then install the CUDA wheels before the other
packages:

```bash
conda create -n cpr-sam3 python=3.12.13 -y
conda activate cpr-sam3
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

This version pair and CUDA index are listed in the
[official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v2100).
The NVIDIA driver must support the selected CUDA runtime. A separate CUDA
toolkit is not needed for the CPU-only MMCV extension build below.

## MMCV

MMSegmentation imports `mmcv.ops`, so the environment needs full MMCV with
`mmcv._ext`, not `mmcv-lite`. The recorded environment used CPU extensions
for this import chain. Neural-network inference remains on CUDA; this does
not provide CUDA implementations of MMCV ops for unrelated models.

Use an MMCV wheel built for the same Python/PyTorch setup if one is available.
An older PyTorch wheel or a CUDA label alone does not establish ABI compatibility.
Otherwise, the following is a Linux source-build route based on the
[MMCV build instructions](https://mmcv.readthedocs.io/en/latest/get_started/build.html)
and [v2.1.0 build configuration](https://github.com/open-mmlab/mmcv/blob/v2.1.0/setup.py):

```bash
# Run from the CPR-SAM3 repository root, after installing PyTorch.
python -m pip install "setuptools<81" wheel ninja psutil "numpy==1.26.4" "mmengine==0.10.7" "opencv-python==4.11.0.86"
git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git third_party/mmcv
CUDA_VISIBLE_DEVICES="" MMCV_WITH_OPS=1 FORCE_CUDA=0 MAX_JOBS=4 \
  python -m pip install --no-build-isolation --no-deps -v ./third_party/mmcv
python -m pip install -r requirements.txt
```

This requires a working C++ compiler. GPU visibility is disabled only for the
build command so MMCV chooses its CPU extension path. `MMCV_WITH_OPS=0` would
produce `mmcv-lite` instead. The setuptools bound retains the `pkg_resources`
API used by MMCV 2.1.0's build script.

The exact original build flags and any toolchain fixes were not recovered.
This source-build route has not been rerun in a fresh Linux environment with
PyTorch 2.10.0; compiler/API compatibility may require additional fixes. The
historical environment record is not a validation of these reconstruction commands.

Install only one OpenCV distribution. MMEngine 0.10.7's published wheel requires
`opencv-python`; adding `opencv-python-headless` would install two packages
that provide the same `cv2` module. On minimal Ubuntu containers, an import
error for `libGL.so.1` requires the system `libgl1` runtime.

## Model Sources

The adapter loads the SAM3 source bundled in the pinned SegEarth-OV3 checkout.
It loads DINOv3 from the directory in `paths.json` using PyTorch Hub's local
source mode. Neither source tree nor either checkpoint is redistributed here.
There is no need to install the full DINOv3 training environment.

## Check the Installation

After configuring `paths.json`, run:

```bash
python -m pip check
python -c "import mmcv._ext; from mmseg.models.data_preprocessor import SegDataPreProcessor; print('MMCV / MMSeg imports OK')"
python tools/doctor.py --paths paths.json
```

`doctor.py` checks dependency availability, compiled MMCV/MMSeg imports, local
assets, the baseline hash, and CUDA availability. It does not load model weights
or run segmentation. A successful check should be followed by a one-image demo
and a benchmark run in the target environment.
