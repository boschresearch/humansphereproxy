## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
from argparse import ArgumentParser
import yaml

import numpy as np
import torch
from tqdm import tqdm

from models.sphere_proxy import SphereProxy
from humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer
from humanml3d_to_smpl.utils.params import beta_hml

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
    dis_all = radii_1 + radii_2
    msk_dists = (dis < dis_all)

    msk_colls = torch.logical_and(msk_dists, msk_tri)

    return msk_colls


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--humanml_dir', help='path to the HumanML3D dataset')
    parser.add_argument("--reg_dir", help="Directory containing the trained regressors and config")
    args = parser.parse_args()

    data_dir = os.path.join(args.humanml_dir, "new_joint_vecs")
    train_ann_file = os.path.join(args.humanml_dir, "train.txt")

    with open(train_ann_file, "r") as train_file:
        lines = train_file.readlines()

    config_path = os.path.join(args.reg_dir,"config_training.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    h2s_layer = HumanML3D_To_SMPL_Layer(smpl_rotation_representation='rot_mat').to(device)
    sp = SphereProxy(args.reg_dir).to(device)

    bw = sp.get_buffer("blend_weights")

    _, assigns_multijoint = torch.topk(bw, 4, dim=-1)
    msk_multijoint1 = (assigns_multijoint[:,0].unsqueeze(1) == assigns_multijoint[:,0])
    msk_multijoint1 = msk_multijoint1.to(device)

    smpl_shape = torch.tensor([beta_hml]).to(device)

    coll_id_mat = torch.zeros((config['model']['num_spheres'],config['model']['num_spheres'])).to(device)
    num_poses = 0
    for line in tqdm(lines):
        # Same motion just mirrored -> will not give any new collision insides
        if line.startswith('M'):
            continue
        hml_path = os.path.join(data_dir, line[:-1]+".npy")
        hml_vec = torch.from_numpy(np.load(hml_path)).permute(1,0).unsqueeze(0).unsqueeze(2).to(device)
        smpl_vec = h2s_layer(hml_vec)

        root_pose = smpl_vec['root_orient'].flatten(0,1)
        body_pose = smpl_vec['body_orient'].flatten(0,1)
        pose = torch.concat((root_pose.unsqueeze(1), body_pose), dim=1)

        smpl_shape_b = smpl_shape.expand(pose.shape[0],-1)
        sp(smpl_shape_b)
        sp.pose_spheres(pose)

        coll_spheres = calc_coll_spheres(sp.posed_centers, sp.radii)
        num_poses += coll_spheres.shape[0]
        coll_id_mat += torch.sum(coll_spheres, dim=0)

    np.save(os.path.join(args.reg_dir, "coll_id_mat.npy"), coll_id_mat.detach().cpu().numpy())
    np.save(os.path.join(args.reg_dir, "num_poses.npy"), num_poses)

    coll_90p = (coll_id_mat >= 0.9*num_poses)
    coll_mj1pd = torch.logical_or(msk_multijoint1, coll_90p)


    np.save(os.path.join(args.reg_dir, "coll_mat_1j.npy"), msk_multijoint1.detach().cpu().numpy())
    np.save(os.path.join(args.reg_dir, "coll_mat_1jd90.npy"), coll_mj1pd.detach().cpu().numpy())