## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
import numpy as np
import torch

from argparse import ArgumentParser

from models.sphere_proxy import SphereProxy
from models.joint_regressor import JointRegressor
from models.sphere_regressor import SphereDecoder

from humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("motion_file", help=".npy file containing the motion to be visualized")
    parser.add_argument("sphere_model", help="Path to the model checkpoint of the sphere regressor")
    parser.add_argument("boneweight_file", help="Path to the model checkpoint of the boneweight regressor")
    parser.add_argument("joint_model", help="Path to the model checkpoint of the joint regressor")
    args = parser.parse_args()

    shape = torch.Tensor([[0.7968, 0.1291, -0.4157, 0.7031, -0.3302, 0.1737, 0.2258, 0.2141, -0.5803, 0.6414]]).to(device)
    kin_tree = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    smpl_shape_dim = 10
    num_joints = 22
    latent_dim = 512
    num_spheres = 192
    hml_motion = torch.from_numpy(np.load(args.motion_file)).to(device)
    hml_motion = hml_motion.transpose(0,1)
    hml_motion = hml_motion.unsqueeze(0)
    hml_motion = hml_motion.unsqueeze(2)

    hml_to_smpl = HumanML3D_To_SMPL_Layer(smpl_rotation_representation='rot_mat').to(device)

    smpl_motion = hml_to_smpl(hml_motion)
    #smpl_motion = smpl_motion.detach().cpu().numpy()

    jr = JointRegressor(smpl_shape_dim, num_joints, latent_dim).to(device)
    jr.load_state_dict(torch.load(args.joint_model))

    sr = SphereDecoder(num_spheres, smpl_shape_dim, latent_dim).to(device)
    sr.load_state_dict(torch.load(args.sphere_model))

    #br = BoneweightRegressor(num_spheres, smpl_shape_dim, latent_dim, num_joints).to(device)
    #br.load_state_dict(torch.load(args.boneweight_model))
    #sr = SphereDecoderWeights(num_spheres, smpl_shape_dim, latent_dim, num_joints).to(device)
    #sr.load_state_dict(torch.load(args.sphere_model))
    #sm = torch.nn.Softmax(dim=2)

    joints = jr(shape).reshape(-1, 22, 3)
    spheres = sr(shape)
    #boneweights = br(shape)
    #spheres, boneweights = sr(shape)
    #boneweights = sm(boneweights)
    boneweights = torch.from_numpy(np.load(args.boneweight_file)).to(device)

    sp = SphereProxy(spheres[:,:,1:], torch.exp(spheres[:,:,0]), joints, kin_tree, boneweights)

    outdir = "8nn_hml_bw_fixed"
    os.makedirs(outdir, exist_ok=True)

    for i in range(smpl_motion['root_orient'].shape[1]):
        root_orient = smpl_motion['root_orient'][:,i,...].unsqueeze(1)
        body_pose = smpl_motion['body_orient'][:,i,...]
        pose = torch.concat((root_orient, body_pose), dim=1)
        trans = smpl_motion['trans'][:,i,...]
        sp.pose_spheres(pose, trans)

        sp.save_as_mesh(os.path.join(outdir, f"{i:05d}.obj"))

