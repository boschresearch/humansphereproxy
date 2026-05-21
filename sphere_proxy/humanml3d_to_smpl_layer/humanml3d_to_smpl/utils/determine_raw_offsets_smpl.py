## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import numpy as np
from humanml3d_to_smpl.utils.params import *
from humanml3d_to_smpl.utils.quaternions import * 
import smplx

if __name__ == "__main__":
    neutral_bm_path = "./body_models/smpl_merged/SMPLH_NEUTRAL.pkl"
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])

    n_bm = smplx.SMPLHLayer(neutral_bm_path)
    b = n_bm(betas=torch.tensor([beta_kit]))

    locs_smpl = b.joints[:,:22].detach().cpu().numpy()

    # The following snippet is from HumanML3D
    #   (https://github.com/EricGuo5513/HumanML3D)
    # Copyright (c) 2022 Chuan Guo, licensed under the MIT license,
    # cf. 3rd-party-licenses.txt file in the root directory of this source tree.
    '''Put on Floor'''
    floor_height = locs_smpl.min(axis=0).min(axis=0)[1]
    locs_smpl[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = locs_smpl[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    locs_smpl = locs_smpl - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = quat_between_points_np(forward_init, target)
    root_quat_init = np.ones(locs_smpl.shape[:-1] + (4,)) * root_quat_init

    locs_smpl = rotate_points_by_quaternion_np(root_quat_init, locs_smpl)
    #locs_smpl = locs_smpl @ trans_matrix
    #locs_smpl[...,0] *= -1


    raw_offsets = np.zeros((22,3))
    for i in range(1, 22):
        off = locs_smpl[0,i] - locs_smpl[0,t2m_parents[i]]
        raw_offsets[i] = off/np.linalg.norm(off)

    print(np.linalg.norm(raw_offsets, axis=1))
    print(raw_offsets.tolist())
