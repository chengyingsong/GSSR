import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from eval import Bconstrast
from data_utils import shuffle
# import torch.fft


class SSFFB(nn.Module):
    def __init__(self, n_feats=64):
        super(SSFFB, self).__init__()
        self.act = nn.ReLU(inplace=True)

        body_spatial = []
        for i in range(2):
            body_spatial.append(
                nn.Conv3d(
                    n_feats,
                    n_feats,
                    kernel_size=(1, 3, 3),
                    stride=1,
                    padding=(0, 1, 1),
                )
            )

        body_spectral = []
        for i in range(2):
            body_spectral.append(
                nn.Conv3d(
                    n_feats,
                    n_feats,
                    kernel_size=(3, 1, 1),
                    stride=1,
                    padding=(1, 0, 0),
                )
            )

        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)
        self.reduce = nn.Conv3d(n_feats * 2, n_feats,
                                kernel_size=(1, 1, 1), stride=1)

    def forward(self, x):
        out = x
        spe = x
        spa = x
        for i in range(2):
            spa = self.body_spatial[i](spa)
            spe = self.body_spectral[i](spe)
            if i == 0:
                spe = self.act(spe)
                spa = self.act(spa)
        out = torch.cat([spe, spa], dim=1)
        out = self.reduce(out)
        # out = spe
        out = out + x
        return out


class Res3DBlock(nn.Module):
    def __init__(self, n_feats=64, bias=True, act=nn.ReLU(True), res_scale=1):
        super(Res3DBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feats, n_feats, (3, 1, 1), 1, (1, 0, 0), bias=bias),
                                  act,
                                  nn.Conv3d(n_feats, n_feats, (1, 3, 3),
                                            1, (0, 1, 1), bias=bias)
                                  )

    def forward(self, x):
        x = self.body((x))+x
        return x


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        TwoTail = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(
                    nn.Conv2d(
                        n_feats,
                        n_feats * 4,
                        kernel_size=(3, 3),
                        stride=1,
                        padding=(1, 1),
                    )
                )
                TwoTail.append(nn.PixelShuffle(2))
        elif scale == 3:
            TwoTail.append(
                nn.Conv2d(
                    n_feats, n_feats * 9, kernel_size=(3, 3), stride=1, padding=(1, 1),
                )
            )
            TwoTail.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*TwoTail)


def _to_4d_tensor(x):
    # B,N,C,H,W
    x = x.permute(0, 2, 1, 3, 4)
    x = torch.split(x, 1, dim=0)
    x = torch.cat(x, 1).squeeze(0)
    return x


def _to_5d_tensor(x, C):
    x = torch.split(x, C)
    x = torch.stack(x, dim=0)
    x = x.transpose(1, 2)
    return x
