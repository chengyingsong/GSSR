from os import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import *


class SGSR(nn.Module):
    def __init__(self, args):
        super(SGSR, self).__init__()

        self.scale = args.upscale_factor
        self.n_feats = args.n_feats
        self.n_module = args.n_module
        self.windSize = args.window_size

        ThreeHead = []
        ThreeHead.append(
            nn.Conv3d(
                1, self.n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
            )
        )
        ThreeHead.append(
            nn.Conv3d(
                self.n_feats,
                self.n_feats,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
            )
        )
        self.ThreeHead = nn.Sequential(*ThreeHead)
        SSFFBs = [SSFFB(self.n_feats) for _ in range(self.n_module)]
        self.SSFFBs = nn.Sequential(*SSFFBs)

        self.reduceD_DFF = nn.Conv2d(
            self.n_feats * self.windSize, self.n_feats, kernel_size=(1, 1), stride=1
        )
        self.conv_DFF = nn.Conv2d(
            self.n_feats, self.n_feats, kernel_size=(1, 1), stride=1
        )

        self.reduceD_FCF = nn.Conv3d(
            self.n_feats * 2, self.n_feats, kernel_size=1, stride=1
        )
        self.conv_FCF = nn.Conv3d(
            self.n_feats, self.n_feats, kernel_size=(1, 1, 1), stride=1
        )
        self.Up = Upsampler(self.scale, args.n_feats)
        self.final = nn.Conv2d(
            self.n_feats, self.windSize, kernel_size=(3, 3), stride=1, padding=1
        )

    def forward(self, x, h=None, i=None):
        # x shape: B,3,H,W --->B,1,3,H,W --> B,N,3,H,W
        y = F.interpolate(x, scale_factor=self.scale, mode="bicubic").clamp(
            min=0, max=1
        )
        x = x.unsqueeze(1)
        x = self.ThreeHead(x)

        skip_x = x
        for j in range(self.n_module):
            x = self.SSFFBs[j](x)

        x = x + skip_x

        # B,N,3,H,W
        if i != 0:
            for j in range(self.windSize):
                x_ilr = torch.cat(
                    [h[:, :, j: j + 1, :, :], x[:, :, j: j + 1, :, :], ], dim=1,
                )  # B,N*2,1,H,W
                x[:, :, j: j + 1, :, :] = self.reduceD_FCF(x_ilr)  # B,N,1,H,W
        x = self.conv_FCF(x)

        h = x
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.reduceD_DFF(x)
        x = self.conv_DFF(x)

        x = self.final(self.Up(x)) + y
        return x, h
