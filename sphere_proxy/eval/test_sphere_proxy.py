## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
from argparse import ArgumentParser

import numpy as np
import torch

import trimesh
import smplx
from tqdm import tqdm

from models.sphere_proxy import SphereProxy
from extensions.mesh2sdf2_cuda import mesh2sdf

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--reg_dir", help="Directory containing the trained regressors and config")
    parser.add_argument("--test_file", help="File containing the test data")

    return parser.parse_args()

def load_data(test_file):
    with open(test_file, 'r') as f:
        filelist = f.readlines()

    # Remove new line characters from end of strings
    filelist = [f.replace("\n", "").replace("joints", "shapes") for f in filelist]

    smpl_shapes = []
    for file in filelist:
        smpl_shapes.append(np.load(file))

    return smpl_shapes

def normalize_meshes(verts):
    # Center mesh around zero
    mesh_max = np.amax(verts, axis=0)
    mesh_min = np.amin(verts, axis=0)
    mesh_center = (mesh_max + mesh_min) / 2
    d_verts = verts - mesh_center

    # Scale mesh to be in [-1,1]
    max_dist = np.sqrt(np.max(np.sum(d_verts**2, axis=-1)))
    mesh_scale = 1.0 / max_dist
    d_verts *= mesh_scale

    return d_verts, mesh_center, mesh_scale
             

if __name__ == "__main__":
    args = get_args()

    # Get the sphere proxy
    sp = SphereProxy(args.reg_dir).to(device)
    smpl_layer = smplx.SMPLHLayer("./body_models/smplh_merged/SMPLH_NEUTRAL.pkl").to(device)

    data = load_data(args.test_file)

    metrics = []
    for shape in tqdm(data):
        shape_t = torch.from_numpy(shape).unsqueeze(0).to(device)

        # Get SMPL mesh
        mesh = smpl_layer(shape_t)
        # Normalize to unit sphere
        norm_verts,_,_ = normalize_meshes(mesh.vertices.squeeze(0).cpu().numpy())

        # Get sphere proxy
        sp(shape_t)
        # Convert to mesh
        sphere_meshes = []
        for i in range(sp.num_spheres):
            tmp = trimesh.creation.icosphere(radius=sp.radii[0,i].detach().cpu().numpy())
            tmp.apply_translation(sp.centers[0,i].detach().cpu().numpy())
            sphere_meshes.append(tmp)
        
        sphere_mesh = trimesh.util.concatenate(sphere_meshes)
        # Normalize to unit sphere
        norm_verts_sp, mesh_sp_center, mesh_sp_scale = normalize_meshes(sphere_mesh.vertices)

        mesh_sp_center_t = torch.from_numpy(mesh_sp_center).unsqueeze(0).to(device)

        sp.centers += mesh_sp_center_t
        sp.centers *= mesh_sp_scale
        sp.radii *= mesh_sp_scale

        # Approximate volume with voxels                    - VOLUME APPRIXIMATION
        prec = 0.02
        x_samples = torch.arange(-1.0, 1.0+prec, prec, device=device)
        x, y, z = torch.meshgrid([x_samples, x_samples, x_samples])
        points = torch.stack([x, y, z], dim=-1).flatten(end_dim=-2)
        sdf, _ = sp.calc_sdf(points)
        a = torch.where(sdf[0] <= 0)[0]
        vol_sp = (len(a)*prec**3)*1e6

        incounts = mesh2sdf.calc_incount(torch.from_numpy(norm_verts[smpl_layer.faces]).unsqueeze(0).to(device), prec)
        b = torch.where(incounts[0] > 0)[0]
        vol_mesh = (len(b)*prec**3)*1e6

        # ALT: Calculate distance of vertices to spheres    - SURFACE APPROXIMATION
        sdf, dists = sp.calc_sdf(torch.from_numpy(norm_verts).unsqueeze(0).to(device))
        sum_sdf = torch.sum(torch.abs(sdf)).item()
        mean_sdf = torch.mean(torch.abs(sdf)).item()

        metrics.append([vol_mesh, vol_sp, sum_sdf])

    metrics = np.array(metrics)

    m_sdf = metrics[:,2]
    m_vol = (metrics[:,0] - metrics[:,1])/metrics[:,0]

    with open(os.path.join(args.reg_dir, "eval_metrics.txt"), "w") as fle:
        fle.write(f"---------- EVALUATION ----------\n")
        fle.write(f"Surface: {np.mean(m_sdf)} +/- {1.96*np.std(m_sdf)/np.sqrt(len(m_sdf))}\n")
        fle.write(f"Volume: {np.mean(m_vol)} +/- {1.96*np.std(m_vol)/np.sqrt(len(m_vol))}\n")

