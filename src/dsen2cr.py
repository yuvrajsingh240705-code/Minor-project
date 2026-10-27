import torch
import torch.nn as nn

class DSen2CR(nn.Module):
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
                f"DSen2CR expects [B,15,H,W] (13 optical + 2 SAR), got {tuple(x.shape)}"
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
