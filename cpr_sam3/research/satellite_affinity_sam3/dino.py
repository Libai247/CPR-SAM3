import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


NORMALIZATION = {
    'lvd': (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    'sat': (
        (0.430, 0.411, 0.296),
        (0.213, 0.156, 0.143),
    ),
}


def _candidate_repositories():
    root = Path(__file__).resolve().parents[2]
    candidates = [
        os.environ.get('DINOV3_REPO', ''),
        root / 'dinov3_server_bundle' / 'dinov3-main',
        Path('third_party')
        / 'dinov3_server_bundle'
        / 'dinov3-main',
    ]
    return [
        str(Path(path).resolve())
        for path in candidates
        if path and Path(path).exists()
    ]


def _resize_divisible(image, max_side, patch_size):
    width, height = image.size
    scale = min(1.0, float(max_side) / max(height, width))
    target_width = max(
        patch_size,
        int(round(width * scale)) // patch_size * patch_size,
    )
    target_height = max(
        patch_size,
        int(round(height * scale)) // patch_size * patch_size,
    )
    if (target_width, target_height) != image.size:
        image = image.resize(
            (target_width, target_height),
            Image.Resampling.BICUBIC,
        )
    return image


def _to_tensor(image, domain, device):
    storage = torch.ByteStorage.from_buffer(image.tobytes())
    tensor = torch.ByteTensor(storage).view(
        image.height,
        image.width,
        3,
    )
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    mean, std = NORMALIZATION[domain]
    mean = tensor.new_tensor(mean).view(3, 1, 1)
    std = tensor.new_tensor(std).view(3, 1, 1)
    return tensor.sub_(mean).div_(std).unsqueeze(0).to(device)


def _load_model(weights, device):
    errors = []
    for repository in _candidate_repositories():
        try:
            model = torch.hub.load(
                repository,
                'dinov3_vitl16',
                source='local',
                pretrained=False,
            )
            state = torch.load(
                str(Path(weights).resolve()),
                map_location='cpu',
                weights_only=True,
            )
            missing, unexpected = model.load_state_dict(
                state,
                strict=False,
            )
            allowed_unexpected = {
                'local_cls_norm.weight',
                'local_cls_norm.bias',
            }
            disallowed = set(unexpected) - allowed_unexpected
            if missing or disallowed:
                raise RuntimeError(
                    'DINO checkpoint mismatch: '
                    f'missing={missing}, unexpected={sorted(disallowed)}'
                )
            model.eval().to(device)
            return model
        except Exception as error:
            errors.append(f'{repository}: {error!r}')
    raise RuntimeError(
        'Unable to load DINOv3 ViT-L: ' + ' | '.join(errors[-3:])
    )


class DINOv3DenseExtractor:
    """Frozen DINOv3-L dense features with domain-correct normalization."""

    patch_size = 16

    def __init__(
        self,
        weights,
        domain,
        device='cuda',
        max_side=512,
        layers=4,
    ):
        if domain not in NORMALIZATION:
            raise ValueError(f'Unknown DINO domain: {domain}')
        self.domain = domain
        self.device = torch.device(device)
        self.max_side = int(max_side)
        self.layers = int(layers)
        self.model = _load_model(weights, self.device)

    @torch.inference_mode()
    def extract(self, image):
        image = _resize_divisible(
            image,
            self.max_side,
            self.patch_size,
        )
        tensor = _to_tensor(image, self.domain, self.device)
        context = (
            torch.autocast('cuda', dtype=torch.float16)
            if self.device.type == 'cuda'
            else torch.no_grad()
        )
        with context:
            outputs = self.model.get_intermediate_layers(
                tensor,
                n=self.layers,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
        features = [
            F.normalize(output[0].float(), dim=0, eps=1e-6)
            for output in outputs
        ]
        last = features[-1]
        multi = F.normalize(
            torch.cat(features, dim=0),
            dim=0,
            eps=1e-6,
        )
        return {'last': last, 'multi': multi}
