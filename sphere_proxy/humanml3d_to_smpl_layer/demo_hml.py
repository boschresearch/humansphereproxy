import os
import torch
import pickle
import smplx

from humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer
from humanml3d_to_smpl.utils.humanml3d_util import plot_3d_motion
from humanml3d_to_smpl.utils.params import hml_kinematic_chain

if __name__ == "__main__":
    import numpy as np
    from datetime import datetime


    smplh_path = "body_models/smplh_merged/SMPLH_NEUTRAL.pkl"
    with open(smplh_path, 'rb') as smplh_file:
        model_data = pickle.load(smplh_file, encoding='latin1')

    smpl_num_verts = model_data['v_template'].shape[0]
    faces = model_data['f']

    # Load hml3d vector and bring in in the format of MDM
    hml3d_vec = torch.from_numpy(np.load("000021.npy")).T.unsqueeze(0).repeat_interleave(8,0).unsqueeze(2).to('cuda')
    bs = hml3d_vec.shape[0]
    frms = hml3d_vec.shape[3]

    s = datetime.now()
    layer_H2S = HumanML3D_To_SMPL_Layer(smpl_rotation_representation="rot_mat").to('cuda')
    layer_SMPL = smplx.SMPLHLayer(smplh_path).to('cuda')
    e = datetime.now()
    print(f"Took {e-s}")


    smpl_params = layer_H2S(hml3d_vec)
    root_orientation = smpl_params['root_orient'].flatten(0,1)
    joint_orientations = smpl_params['body_orient'].flatten(0,1)
    root_location = smpl_params['trans'].flatten(0,1)
    res = layer_SMPL(betas=None,
                    global_orient=root_orientation,
                    body_pose=joint_orientations,
                    transl=root_location)
    verts = res.vertices.reshape((bs, frms,) + res.vertices.shape[1:])
    jnts = res.joints.reshape((bs, frms,) + res.joints.shape[1:])
    jnts = jnts[:,:,:22,...]

    os.makedirs("animation", exist_ok=True)
    plot_3d_motion("animation/000009.mp4", hml_kinematic_chain, jnts[1].detach().cpu().numpy(), "First Test", fps=20)
    print()
