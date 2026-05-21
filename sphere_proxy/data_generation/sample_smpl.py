## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import torch
import smplx
import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation


def sample_smpl(num_samples, range_min=-5, range_max=5, sample_pose=False, max_rot=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smplLayer = smplx.SMPLHLayer("body_models/smplh_merged/SMPLH_NEUTRAL.pkl").to(device)
    beta = torch.rand(num_samples,10).to(device)
    faces = smplLayer.faces.astype(np.int32)


    # Scale values from range [0,1) to desired range
    beta = range_min + (range_max - range_min)*beta

    # The default mesh is always required to obtain joint locations and scale
    default_mesh = smplLayer(betas=beta)

    ### Normalization
    d_verts = default_mesh.vertices
    d_verts = d_verts[:, faces.astype(np.int32)]
    d_verts = d_verts.reshape(num_samples,-1,3).detach().cpu().numpy()

    # Center mesh around zero
    mesh_max = np.amax(d_verts, axis=1)
    mesh_min = np.amin(d_verts, axis=1)
    mesh_center = (mesh_max + mesh_min) / 2
    d_verts = d_verts - mesh_center[:,np.newaxis,...]

    # Scale mesh to be in [-1,1]
    max_dist = np.sqrt(np.max(np.sum(d_verts**2, axis=-1), axis=1))
    mesh_scale = 1.0 / max_dist

    #print(np.sqrt(np.max(np.sum((d_verts*mesh_scale[:,np.newaxis, np.newaxis])**2, axis=-1), axis=1)))


    # Center and scale joints in default pose
    joints = default_mesh.joints[:,:22].detach().cpu().numpy()
    joints -= mesh_center[:,np.newaxis,...]
    joints *= mesh_scale[:,np.newaxis, np.newaxis]

    
    if sample_pose:
        # Sample random rotations and scale the magnitude to the desired range
        rot = Rotation.random(num_samples*22).as_rotvec(degrees=True)
        angles = np.linalg.norm(rot, axis=-1)
        new_angles = np.random.uniform(high=max_rot, size=num_samples*22)
        rot /= angles[...,np.newaxis]
        rot *= new_angles[...,np.newaxis]
        
        pose = Rotation.from_rotvec(rot, degrees=True).as_matrix().reshape(num_samples, 22, 3, 3)
        pose = torch.from_numpy(pose).float().to(device)
        mesh = smplLayer(betas=beta, global_orient=pose[:,0:1], body_pose=pose[:,1:])
    else:
        mesh = default_mesh

    # Get (posed) vertices and apply normalization
    verts = mesh.vertices.detach().cpu().numpy()
    verts -= mesh_center[:,np.newaxis,...]
    verts *= mesh_scale[:,np.newaxis, np.newaxis]
    #verts = verts[:, faces.astype(np.int32)]

    #print(np.sqrt(np.max(np.sum(verts.reshape(num_samples,-1,3)**2, axis=-1), axis=1)))


    poses = torch.cat([mesh.global_orient, mesh.body_pose], dim=1).detach().cpu().numpy()


    return verts, beta.detach().cpu().numpy(), poses, joints

def main(args):
    data_dir = args.out_dir
    print(f"Sample pose: {args.sample_pose}")
    class_id = "0"
    tri_dir = os.path.join(data_dir, "triangles", class_id)
    shape_dir = os.path.join(data_dir, "shapes", class_id)
    pose_dir = os.path.join(data_dir, "poses", class_id)
    joint_dir = os.path.join(data_dir, "joints", class_id)
    os.makedirs(tri_dir, exist_ok=True)
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(joint_dir, exist_ok=True)

    for mb in range(args.num_minibatches):
        mb_samples = args.num_samples//args.num_minibatches
        verts, betas, poses, joints = sample_smpl(mb_samples, sample_pose=args.sample_pose, max_rot=args.max_rot)

        for i in range(mb_samples):
            file_name = f"{i+mb*mb_samples:08d}.npy"
            save_path_tri = os.path.join(tri_dir, file_name)
            save_path_shape = os.path.join(shape_dir, file_name)
            save_path_pose = os.path.join(pose_dir, file_name)
            save_path_joint = os.path.join(joint_dir, file_name)
            np.save(save_path_tri, verts[i])
            np.save(save_path_shape, betas[i])
            np.save(save_path_pose, poses[i])
            np.save(save_path_joint, joints[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', default=10, type=int, help="Number of smpl meshes to generate.")
    parser.add_argument('--out_dir', default="./dataset/smpl_samples/", type=str, help="Directory where samples are saved")
    parser.add_argument('--sample_pose', default=False, action='store_true')
    parser.add_argument('--max_rot', default=30, type=int, help='max rotation in degrees (between 0 and 180)')
    parser.add_argument('--num_minibatches', default=1, type=int)
    args = parser.parse_args()

    main(args)

