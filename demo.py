import numpy as np
import torch
from argparse import ArgumentParser

from sphere_proxy.models.sphere_proxy import SphereProxy
from sphere_proxy.humanml3d_to_smpl_layer.humanml3d_to_smpl.utils.params import *
from sphere_proxy.humanml3d_to_smpl_layer.humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer
from sphere_proxy.eval.metrics import selfintersection_metric

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--reg_dir", help="Directory containing the trained regressors and config")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Get the sphere proxy
    sp = SphereProxy(args.reg_dir).to(device)

    # Set the spheres given a smpl shape
    hml_shape = torch.tensor([beta_hml]).to(device)
    sp(hml_shape)

    # Save the sphere proxy as a mesh
    sp.save_as_mesh("demo_mesh.obj")

    # Pose the sphere proxy
    # Reconstruct the pose from HumanML3D representation; However, you can also
    # directly use smpl joint rotations
    hml_vec = torch.from_numpy(np.load("sphere_proxy/humanml3d_to_smpl_layer/000021.npy")).to(device)
    hml_vec = hml_vec.permute([1,0]).unsqueeze(1).unsqueeze(0)
    hml_to_smpl = HumanML3D_To_SMPL_Layer(smpl_rotation_representation='rot_mat', dataset='hml').to(device)
    smpl_params = hml_to_smpl(hml_vec)
    root_orientation = smpl_params['root_orient'][0]
    joint_orientations = smpl_params['body_orient'][0]

    # Pose the spheres
    pose = torch.concat((root_orientation.unsqueeze(1), joint_orientations), dim=1)
    sp.pose_spheres(pose)

    # Calculate self-intersection loss
    selfintersections = sp.selfintersection_loss()
    loss = torch.mean(selfintersections)
    loss.backward()
    print(loss)

    # Calculate SI metric for the demo motion
    metric = selfintersection_metric(hml_vec.to(device), smpl_path="sphere_proxy/body_models/smplh_merged/SMPLH_NEUTRAL.pkl")
    print(metric.mean())