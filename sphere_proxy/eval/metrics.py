## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch

from sphere_proxy.humanml3d_to_smpl_layer.humanml3d_to_smpl.utils.params import beta_hml, beta_kit
from sphere_proxy.humanml3d_to_smpl_layer.humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer
from sphere_proxy.extensions.mesh2sdf2_cuda import mesh2sdf
from sphere_proxy.util.util import get_mesh_normalization

import smplx


def selfintersection_metric(motion, smpl_path, smpl_shape_params=None, prec=0.06, dataset_name="hml"):
    if smpl_shape_params is None:
        if dataset_name == "hml":
            smpl_shape_params = torch.tensor([[beta_hml]]).to(motion.device)
            smpl_shape_params = smpl_shape_params.expand(motion.shape[0], motion.shape[-1], -1)
        elif dataset_name == "kit":
            smpl_shape_params = torch.tensor([[beta_kit]]).to(motion.device)
            smpl_shape_params = smpl_shape_params.expand(motion.shape[0], motion.shape[-1], -1)
    
    # 1. Recover SMPL parameters
    hml_to_smpl = HumanML3D_To_SMPL_Layer(smpl_rotation_representation='rot_mat',
                                        dataset=dataset_name).to(motion.device)
    smpl_params = hml_to_smpl(motion.float())

    # 2. Get meshes
    # Merge batch size and number of frames
    root_orientation = smpl_params['root_orient'].flatten(0,1)
    joint_orientations = smpl_params['body_orient'].flatten(0,1)
    root_location = smpl_params['trans'].flatten(0,1)
    shape = smpl_shape_params.flatten(0,1)

    # 3. Calc selfintersections
    # Normalize meshes to be in unit sphere
    smpl_layer = smplx.SMPLHLayer(smpl_path).to(motion.device)

    meshes = smpl_layer(betas=shape)
    mc, ms = get_mesh_normalization(meshes, smpl_layer.faces.astype(np.int32))

    # Don't use translation as it is not relevant for self-intersections
    meshes = smpl_layer(betas=shape,
            global_orient=root_orientation,
            body_pose=joint_orientations)

    verts = meshes.vertices - mc
    verts *= ms

    si_metric, _ = calculate_selfintersection_volume(verts, smpl_layer.faces.astype(np.int32))
    si_metric = si_metric.reshape(motion.shape[0], motion.shape[-1])

    return si_metric
    

def calculate_selfintersection_volume(vertices, faces, prec=0.06):
    """ Given triangular meshes, approximate the self-intersection volume
        using voxels. Assumes that the mesh is centered at zero and scaled so
        they are within a unit sphere.
        
        Params:
            - vertices: torch.Tensor (b, num_verts, 3)
            - faces: torch.Tensor (num_faces, 3)
            - prec: side length of a voxel
        Returns:
            - vols_met: (b) self-intersection volume in cm^3
            - vols_perc: (b) self-intersection volume in percent relative
                             to original volume 


    """
    # Determine which voxel is inside a self-intersection area
    incounts = mesh2sdf.calc_incount(vertices[:,faces], prec)

    # Approximate original mesh volume
    vols = (torch.sum(incounts, dim=1)*prec**3)*1e6

    # Approximate self-intersection volume
    incounts[incounts<2] = 0
    vols_is = (torch.sum(incounts, dim=1)*prec**3)*1e6

    return vols_is, 100*vols_is/vols