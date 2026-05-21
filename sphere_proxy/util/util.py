## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import argparse
import yaml
import torch
import numpy as np
import random


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                "config",
                help="Config file containing training settings"
                )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, args.config


def get_mesh_normalization(default_meshes, faces):
    """
        The following snippet is from DualSDF
        (https://github.com/zekunhao1995/DualSDF/blob/master/sample_sdfs.py)
        Copyright (c) 2020 Zekun Hao, licensed under the MIT license,
        cf. 3rd-party-licenses.txt file in the root directory of this source tree.
    """

    ### Normalization
    d_verts = default_meshes.vertices.detach().cpu()
    d_verts = d_verts[:, faces.astype(np.int32)]
    d_verts = d_verts.reshape(len(d_verts),-1,3)

    # Center mesh around zero
    mesh_max = torch.amax(d_verts, axis=1)
    mesh_min = torch.amin(d_verts, axis=1)
    mesh_center = (mesh_max + mesh_min) / 2
    d_verts = d_verts - mesh_center.unsqueeze(1)

    # Scale mesh to be in [-1,1]
    max_dist = torch.sqrt(torch.amax(torch.sum(d_verts**2, dim=-1), dim=1))
    mesh_scale = 1.0 / max_dist

    return mesh_center.unsqueeze(1).to(default_meshes.vertices.device), mesh_scale.unsqueeze(-1).unsqueeze(-1).to(default_meshes.vertices.device)


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)