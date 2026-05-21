## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
from argparse import ArgumentParser
import numpy as np
import torch
from humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer
from humanml3d_to_smpl.utils import params
#from model.sphere_regressor_decoder import SphereDecoder
#from model.joint_regressor import JointRegressor
#from model.sphere_proxy import SphereProxy
import smplx
from extensions.mesh2sdf2_cuda import mesh2sdf
import trimesh
import yaml
from util.util import get_mesh_normalization
from visualize.create_video import calculate_selfintersection_volume

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_coll_spheres(cntrs, radii):
    num_spheres = cntrs.shape[1]
    sph_idx = torch.arange(num_spheres)
    msk_tri = (sph_idx[:,None] < sph_idx).unsqueeze(0).to(device)

    radii_1 = radii.unsqueeze(-1)
    radii_2 = radii.unsqueeze(-2)
    cntrs_1 = cntrs.unsqueeze(-2)
    cntrs_2 = cntrs.unsqueeze(-3)

    diff = cntrs_1 - cntrs_2
    dis = torch.norm(diff, dim=-1)
    dis_all = (radii_1 + radii_2)*0.95
    msk_dists = (dis < dis_all)

    msk_colls = torch.logical_and(msk_dists, msk_tri)

    return msk_colls

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--num_frames", default=7, type=int)
    parser.add_argument("--motion_path", type=str)
    parser.add_argument("--out_dir", default="./visualize_objs", type=str)
    parser.add_argument("--sphere_proxy_path", type=str)
    parser.add_argument("--motion_id", default=0, type=int)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    smpl_layer = smplx.SMPLHLayer("./body_models/smplh_merged/SMPLH_NEUTRAL.pkl").to(device)
    h2s_layer = HumanML3D_To_SMPL_Layer(smpl_rotation_representation="rot_mat").to(device)

    motion = np.load(args.motion_path, allow_pickle=True)[()]['hml'][args.motion_id:args.motion_id+1]
    #motion = torch.from_numpy(motion).unsqueeze(-1).unsqueeze(0)
    print(motion.shape)
    motion = torch.from_numpy(motion).float().to(device).permute([0,2,3,1])
    #motion = motion.to(device).permute([0,2,3,1])
    vis_idxs = np.linspace(0,motion.shape[3]-1,args.num_frames, dtype=int)

    smpl_shape = torch.tensor([params.beta_hml]).expand([motion.shape[3],-1]).to(device)

    meshes = smpl_layer(betas=smpl_shape)
    mc, ms = get_mesh_normalization(meshes, smpl_layer.faces.astype(np.int32))

    smpl_params = h2s_layer(motion)

    ### Sphere Proxy Setup
    # with open(os.path.join(args.sphere_proxy_path, "config_training.yaml"), 'r') as f:
    #     sp_cfg = yaml.safe_load(f)
    # joint_reg = JointRegressor(sp_cfg["model"]["smpl_shape_dim"],
    #                                 sp_cfg["model"]["num_joints"],
    #                                 sp_cfg["model"]["latent_dim"]).to(device)

    # joint_reg.load_state_dict(torch.load(os.path.join(args.sphere_proxy_path, "JR", "joint_regressor.pth")))
    # joint_reg.eval()

    # sphere_reg = SphereDecoder(sp_cfg["model"]["num_spheres"],
    #                                 sp_cfg["model"]["smpl_shape_dim"],
    #                                 sp_cfg["model"]["latent_dim"],
    #                                 sp_cfg["model"]["num_joints"]).to(device)
    
    # sphere_reg.load_state_dict(torch.load(os.path.join(args.sphere_proxy_path, "SR", "sphere_regressor.pth")))
    # sphere_reg.eval()

    # boneweights = torch.from_numpy(np.load(os.path.join(args.sphere_proxy_path, "boneweights.npy"))).unsqueeze(0).to(device)
    # boneweights.requires_grad = True

    # kin_tree = np.array(params.t2m_parents)

    # # Ensure that only upper triangle matrix is used
    # sph_idx = torch.arange(sp_cfg["model"]["num_spheres"])
    # coll_mask = (sph_idx[:,None] < sph_idx)

    # # Invert to get the indices to check for collision
    # coll_mat = ~torch.from_numpy(np.load(os.path.join(args.sphere_proxy_path, "coll_mat_1jd90.npy")))
    # coll_mask = torch.mul(coll_mask, coll_mat)


    # coll_idx = torch.where(coll_mask) 

    # spheres = sphere_reg(smpl_shape)
    # joints = joint_reg(smpl_shape).reshape(-1, sp_cfg["model"]["num_joints"], 3)
    # sp = SphereProxy(spheres[...,1:], torch.exp(spheres[...,0]), joints, kin_tree, boneweights)

    # # Pose spheres
    # pose = torch.concat((smpl_params['root_orient'].flatten(0,1).unsqueeze(1), smpl_params['body_orient'].flatten(0,1)), dim=1)
    # sp.pose_spheres(pose, 10*smpl_params['trans'].flatten(0,1).unsqueeze(1))   
    ### Sphere Proxy Setup

    smpl_meshes = smpl_layer(betas=smpl_shape,
                             global_orient=smpl_params['root_orient'].flatten(0,1),
                             body_pose=smpl_params['body_orient'].flatten(0,1))

    verts = smpl_meshes.vertices - mc
    verts *= ms

    si_metric_mb, _ = calculate_selfintersection_volume(verts, smpl_layer.faces.astype(np.int32))
    sis = np.mean(si_metric_mb)
    
    faces = torch.tensor(smpl_layer.faces.astype(np.int64),
                         dtype=torch.long,
                         device=device)
    
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "si.txt"), 'w+') as fl:
        fl.write(f"SI-Metric: {sis}")

    # Save meshes
    for i in vis_idxs:
        vrts = smpl_meshes.vertices[i]#.unsqueeze(0)

        surface_triangles = mesh2sdf.trimmesh_gpu(vrts[faces])

        surface_ids = torch.where(surface_triangles)[0].cpu().numpy()
        inside_ids = torch.where(~surface_triangles)[0].cpu().numpy()

        vrts_tri = vrts + smpl_params['trans'].flatten(0,1)[i]

        mesh_out = trimesh.Trimesh(vrts_tri.squeeze(0).cpu().numpy(), faces[surface_ids].cpu().numpy())
        mesh_in = trimesh.Trimesh(vrts_tri.squeeze(0).cpu().numpy(), faces[inside_ids].cpu().numpy())
        mesh_out.export(os.path.join(args.out_dir, f"MeshOut_{i:04d}.obj"))
        mesh_in.export(os.path.join(args.out_dir, f"MeshIn_{i:04d}.obj"))

    
    # Save spheres
    # for i in vis_idxs:
    #     frm_colls = calc_coll_spheres(sp.posed_centers[i].unsqueeze(0), sp.radii[i])[0].cpu()

    #     loss_spheres = torch.logical_and(frm_colls, coll_mat)

    #     colliding_spheres = set(torch.concat(torch.where(loss_spheres)).tolist())
    #     uncolliding_spheres = set(sph_idx.tolist()).difference(colliding_spheres)

    #     coll_mesh = []
    #     for idx in colliding_spheres:
    #         tmp = trimesh.creation.icosphere(radius=sp.radii[i,idx].detach().cpu().numpy())
    #         tmp.apply_translation(sp.posed_centers[i, idx].detach().cpu().numpy())
    #         coll_mesh.append(tmp)

    #     uncoll_mesh = []
    #     for idx in uncolliding_spheres:
    #         tmp = trimesh.creation.icosphere(radius=sp.radii[i,idx].detach().cpu().numpy())
    #         tmp.apply_translation(sp.posed_centers[i, idx].detach().cpu().numpy())
    #         uncoll_mesh.append(tmp)

    #     coll_mesh = trimesh.util.concatenate(coll_mesh)
    #     uncoll_mesh = trimesh.util.concatenate(uncoll_mesh)

    #     coll_mesh.export(os.path.join(args.out_dir, f"SphereIn_{i:04d}.obj"))
    #     uncoll_mesh.export(os.path.join(args.out_dir, f"SphereOut_{i:04d}.obj"))

