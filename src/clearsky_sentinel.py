import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Mapping

class ClearSkySentinel(nn.Module):
    """
    Inputs:
      - 13 Sentinel-2 channels in B01..B12/B8A order used by SEN12MS-CR
      - 2 Sentinel-1 channels (VV, VH)
    Architecture:
      15-channel input -> 256-channel conv -> 16 residual blocks
      -> 13-channel conv -> long skip from the cloudy optical input.
    """

    def __init__(self, features=256, blocks=16, res_scale=0.1):
        super().__init__()

        self.inp = nn.Sequential(
            nn.Conv2d(15, features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [ResBlock(features, res_scale=res_scale) for _ in range(blocks)]
        )

        self.out = nn.Conv2d(features, 13, 3, padding=1, bias=True)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 15:
            raise ValueError(
                f"ClearSkySentinel expects [B,15,H,W] (13 optical + 2 SAR), got {tuple(x.shape)}"
            )

        ms = x[:, :13]

        z = self.inp(x)
        for b in self.blocks:
            z = b(z)

        return ms + self.out(z)


class ResBlock(nn.Module):
    def __init__(self, dim, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.c2(y)
        return x + self.res_scale * y


def load_clearsky_weights(
    model: ClearSkySentinel,
    checkpoint_path: str | Path,
    *,
    map_location=None,
) -> ClearSkySentinel:
    """Load a raw or wrapped ClearSky-AI Sentinel state dictionary into a model."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{path} is a Git LFS pointer, not a model checkpoint. "
                "Run 'git lfs pull' and retry."
            )

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Unsupported checkpoint format in {path}: expected a state dictionary or checkpoint mapping."
        )

    state_dict = None
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state_dict = candidate
            break
    if state_dict is None:
        state_dict = checkpoint

    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid state-dictionary key in {path}: {key!r}")
        normalized_key = key
        while normalized_key.startswith(("module.", "model.")):
            normalized_key = normalized_key.split(".", 1)[1]
        normalized[normalized_key] = value

    if not normalized:
        raise ValueError(f"No tensor weights were found in checkpoint {path}.")

    model.load_state_dict(normalized, strict=True)
    return model
