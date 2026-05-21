## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
import torch
import smplx
import numpy as np
from tqdm import tqdm

from models.sphere_regressor import SphereRegressor
from util.util import get_config


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_rest_colls(centers, radii):
    radii_1 = radii.unsqueeze(-1)
    radii_2 = radii.unsqueeze(-2)
    cntrs_1 = centers.unsqueeze(-2)
    cntrs_2 = centers.unsqueeze(-3)

    diff = cntrs_1 - cntrs_2
    dis = torch.norm(diff, dim=-1)
    dis_all = radii_1 + radii_2
    colls = dis - dis_all
    # # Set diagonal to zero as these spheres will always collide
    colls.diagonal(dim1=-1, dim2=-2).zero_()
    # # Matrices are symmetric, we only need one half
    colls = torch.triu(colls)

    idxs = colls > 0
    return idxs


def calc_boneweights(centers, radii, vertices, smpl_boneweights, num_n):
    dist_to_verts = torch.norm(centers.unsqueeze(1) - vertices, dim=-1)
    sur_dists = dist_to_verts - radii.unsqueeze(-1)

    nn_vals, nearest_neighs = torch.topk(sur_dists, num_n, dim=1, largest=False)

    nn_boneweights = smpl_boneweights[nearest_neighs]
    nn_vals /= torch.sum(nn_vals, dim=-1, keepdim=True)

    nn_w_boneweights = torch.mul(nn_boneweights, nn_vals.unsqueeze(-1))

    sph_boneweights = torch.sum(nn_w_boneweights, dim=1)

    return sparseify(sph_boneweights)

def reduce_smpl_boneweights(smpl_boneweights):
    reduced_weights = torch.zeros((smpl_boneweights.shape[0], 22)).to(smpl_boneweights.device)
    reduced_weights[:,:22] = smpl_boneweights[:,:22]
    reduced_weights[:,20] += torch.sum(smpl_boneweights[:,22:37], dim=-1)
    reduced_weights[:,21] += torch.sum(smpl_boneweights[:,37:], dim=-1)

    return reduced_weights

def sparseify(boneweights):
    s_weights = torch.zeros_like(boneweights)
    vs, idxs = torch.topk(boneweights, 4, dim=-1)
    for i in range(idxs.shape[0]):
        for j in range(idxs.shape[1]):
            s_weights[i,idxs[i,j]] = vs[i,j]

    s_weights /= torch.sum(s_weights, dim=-1, keepdim=True)

    return s_weights

if __name__ == "__main__":
    cfg, cfg_path = get_config()
    save_dir = "/".join(cfg_path.split("/")[:-1])

    sphere_checkpoint_path = os.path.join(save_dir, "SR", "sphere_regressor.pth")
    smpl_path = "body_models/smplh_merged/SMPLH_NEUTRAL.pkl"

    smpl_layer = smplx.SMPLHLayer(smpl_path).to(device)
    sphere_reg = SphereRegressor(cfg['model']['num_spheres'],
                               cfg['model']['smpl_shape_dim'],
                               cfg['model']['latent_dim']).to(device)
    sphere_reg.load_state_dict(torch.load(sphere_checkpoint_path, weights_only=True))
    sphere_reg.eval()
    sphere_reg = sphere_reg.to(device)

    num_n = cfg['model']['boneweights_nn']

    smpl_weights = reduce_smpl_boneweights(smpl_layer.lbs_weights)

    num_shape_samples = 1000
    weights = []
    for i in tqdm(range(num_shape_samples)):
        shape = (torch.rand(10)-0.5)*10
        shape = shape.unsqueeze(0).to(device)

        mesh = smpl_layer(betas=shape)

        verts = mesh.vertices
        verts = verts.reshape(1,-1,3)
        mesh_max,_ = torch.max(verts, dim=1)
        mesh_min,_ = torch.min(verts, dim=1)
        mesh_center = (mesh_max + mesh_min) / 2
        verts = verts - mesh_center[:,np.newaxis,...]
        max_dist = torch.sqrt(torch.max(torch.sum(verts**2, dim=-1), dim=1)[0])
        mesh_scale = 1.0 / max_dist
        verts *= mesh_scale

        spheres = sphere_reg(shape)
        cntrs = spheres[0,:,1:]
        rad = torch.exp(spheres[0,:,0])

        weights.append(calc_boneweights(cntrs, rad, verts, smpl_weights, num_n).detach().cpu().numpy())

    weights = torch.from_numpy(np.array(weights))
    m_weights = torch.mean(weights, axis=0)
    std_weights = torch.std(weights, axis=0)
    max_std = torch.topk(std_weights.flatten(), k=8)
    mean_std = torch.mean(std_weights)

    m_weights = sparseify(m_weights)
    np.save(os.path.join(save_dir, "boneweights.npy"), m_weights)
