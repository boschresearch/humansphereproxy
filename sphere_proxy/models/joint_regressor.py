## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

from torch import nn


class JointRegressor(nn.Module):
    """Regress from SMPL shape parameter to SMPL joint locations. This is necessary because
       SMPL determines the joint locations using the vertex locations and this uses to 
       much memory for our use case."""
    def __init__(self, smpl_shape_dim, num_joints, latent_dim):
        super().__init__()

        self.smpl_shape_dim = smpl_shape_dim
        self.num_joints = num_joints
        self.latent_dim = latent_dim

        self.linear_stack = nn.Sequential(
            nn.Linear(self.smpl_shape_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, 3*self.num_joints)
        )


    def forward(self, x):
        return self.linear_stack(x)


