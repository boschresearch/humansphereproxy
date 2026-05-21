import os
import torch
import pickle
import smplx
from zipfile import ZipFile
import trimesh

from humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer
from humanml3d_to_smpl.utils.humanml3d_util import plot_3d_motion
from humanml3d_to_smpl.utils.params import hml_kinematic_chain, kit_kinematic_chain, beta_kit

if __name__ == "__main__":
    import numpy as np
    from datetime import datetime

    hml3d_vec = np.load("/home/hpr2hi/data/KIT-ML/new_joints/03950.npy")
    plot_3d_motion("animation/KIT_03950.mp4", kit_kinematic_chain, hml3d_vec, "First Test", fps=15, radius=2000)
    shape = torch.Tensor([beta_kit]).to('cuda')



    smplh_path = "body_models/smpl_merged/SMPLH_NEUTRAL.pkl"
    with open(smplh_path, 'rb') as smplh_file:
        model_data = pickle.load(smplh_file, encoding='latin1')

    smpl_num_verts = model_data['v_template'].shape[0]
    faces = model_data['f']

    # Load hml3d vector and bring in in the format of MDM
    hml3d_vec = torch.from_numpy(np.load("03950.npy")).type(torch.float32).T.unsqueeze(0).repeat_interleave(8,0).unsqueeze(2).to('cuda')
    bs = hml3d_vec.shape[0]
    frms = hml3d_vec.shape[3]

    s = datetime.now()
    layer_H2S = HumanML3D_To_SMPL_Layer(smpl_rotation_representation="rot_mat", dataset='kit').to('cuda')
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

    plot_3d_motion("animation/KIT_03950_rec.mp4", hml_kinematic_chain, jnts[1].detach().cpu().numpy(), "First Test", fps=15)

    
    with ZipFile("KIT-RECONSTRUCT.zip", "w") as zip_object:
        for i, m in enumerate(hml3d_vec[0:1]):
            m = m.unsqueeze(0)
            smpl_params = layer_H2S(m)
            go = smpl_params['root_orient'].squeeze(0)
            bp = smpl_params['body_orient'].squeeze(0)
            tr = smpl_params['trans'].squeeze(0)
            sh = shape.expand([bp.shape[0], -1])
            meshes = layer_SMPL(betas=sh, global_orient=go, body_pose=bp, transl=tr)

            for frm in range(go.shape[0]):
                mesh = trimesh.Trimesh(vertices=meshes.vertices[frm].detach().cpu().numpy(), faces=layer_SMPL.faces)
                with zip_object.open(os.path.join(f"KIT-RECONSTRUCT_{i}", f"frame{frm:03d}.obj"), "w") as obj_file:
                    mesh.export(obj_file, "obj")

