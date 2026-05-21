## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

## This source code is derived from DualSDF @ e43241c28e20eefb78ca7f467274bb56168dc1f0
##   (https://github.com/zekunhao1995/DualSDF)
## Copyright (c) 2020 Zekun Hao, licensed under the MIT license,
## cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
from torch import nn


class SphereRegressor(nn.Module):
    """Architecture following DualSDF. Instead of using a Gaussian latent space,
       we use the SMPL shape parameters as latent variables. So it is basically
       just a regressor from SMPL to sphere parameters."""
    def __init__(self, num_spheres, smpl_shape_dim, latent_dim):
        super().__init__()
        self.num_spheres = num_spheres
        self.smpl_shape_dim = smpl_shape_dim
        self.latent_dim = latent_dim

        input_dim = self.smpl_shape_dim
        output_dim = 4*self.num_spheres

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim, self.latent_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(self.latent_dim, self.latent_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(self.latent_dim, self.latent_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(self.latent_dim, self.latent_dim - input_dim)),
            nn.ReLU(inplace=True)
        )

        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(self.latent_dim, self.latent_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(self.latent_dim, self.latent_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(self.latent_dim, self.latent_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(self.latent_dim, self.latent_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, output_dim)
        )


    def forward(self, x):
        in1 = x
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)

        # Reshape to sphere params
        # [log(r), center_x, center_y, center_z]
        out2 = out2.reshape(-1, self.num_spheres, 4)
        return out2

