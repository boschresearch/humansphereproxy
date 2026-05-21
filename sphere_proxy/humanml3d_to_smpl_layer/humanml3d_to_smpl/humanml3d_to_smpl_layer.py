## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import torch
from torch import nn


from humanml3d_to_smpl.utils.humanml3d_util import *
from humanml3d_to_smpl.utils.actor_util import *
from humanml3d_to_smpl.utils.quaternions import *
from humanml3d_to_smpl.utils.params import *

class HumanML3D_To_SMPL_Layer(nn.Module):
    def __init__(self,
                 smpl_rotation_representation = 'ax_an',
                 dataset='hml'):
        super().__init__()


        if smpl_rotation_representation in ['ax_an', 'rot_mat']:
            self.smpl_rotation_representation = smpl_rotation_representation
        else:
            raise Exception("[HumanML3D_To_SMPL_Layer] rotation representation not supported")
        
        if dataset == 'hml' or dataset == 'humanml':
            self.dataset = dataset
            self.num_joints = 22
            self.register_buffer('smpl_raw_offsets', torch.tensor(hml_raw_offsets))
        elif dataset == 'kit':
            self.dataset = dataset
            self.num_joints = 21
            self.register_buffer('smpl_raw_offsets', torch.tensor(kit_raw_offsets))
        else:
            raise Exception("[HumanML3D_To_SMPL_Layer] dataset not supported")



        self.kinematic_chain = hml_kinematic_chain
        self.face_joint_idx = face_joint_indx



        self._parents = [0] * len(self.smpl_raw_offsets)
        self._parents[0] = -1
        for chain in self.kinematic_chain:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]
        self._children = {}
        for jnt in range(len(self._parents)):
            self._children[jnt] = []
            for chld in range(len(self._parents)):
                if jnt == self._parents[chld]:
                    self._children[jnt].append(chld)


    def forward(self, data):
        """
        Data format:
            - 1 root angular velocity around y axis                     (index 0 - index 1)
            - 1 root linear velocity in x direction                     (index 1 - index 2)
            - 1 root linear velocity in z direction                     (index 2 - index 3)
            - 1 root height                                             (index 3 - index 4)
            - (num_joints-1)*3 local joint positions                    (index 4 - index 67)
            - (num_joints-1)*6 local joint rotations                    (index 67 - index 193)
            - (num_joints)*3 local velocities                           (index 193 - index 259)
            - 4 foot ground contact features                            (index 259 - index 263)

        Params:
        data (Tensor (batch_size x njoints x nfeats x frames)): pose parameters in HumanML3D format            # MDM standard: 64 x 263 x 1 x 196  
        is_normalized (bool): determine if data is normalized or not
        """
        # Data is already in joint locations, only need to recover joint rotations
        if data.shape[1] == 22:
            positions = data.permute([0,3,1,2])
        else:
            if data.shape[2] != 1:
                raise Exception("HumanML3D_To_SMPL_Layer: njoints is supposed to be 1 I guess")
            else:
                data = data.squeeze(2)

            # Change dimension order. Last one has to be HumanML3D features
            data = data.permute([0, 2, 1])

            positions = recover_from_ric(data, self.num_joints)

        # KIT has a different skeletal structure than SMPL, but many joints
        # correspond to one another. KIT is missing spine2, left_collar, and
        # right_collar and has additionally left_heel and right_heel. Here,
        # we discard the extra joints, reconstruct the missing joints from
        # the existing ones and reorder the joints so they have the SMPL 
        # structure
        if self.dataset == 'kit':
            # KIT seems to use cm, HumanML3D uses m
            #positions /= kit_scale
            # We also need to flip along the facing direction
            positions[:,:,:,0] *= -1 

            # Mapping from KIT joints to SMPL joints
            correspondence = [0,11,16,1,12,17,13,18,2,15,20,3,4,5,8,6,9,7,10]
            old_pos = positions[:,:,correspondence]

            # Reconstruct missing joints
            spine2 = (positions[:,:,1:2] + positions[:,:,2:3])/2.
            left_collar = (positions[:,:,2:3]+positions[:,:,5:6])/2.
            right_collar = (positions[:,:,2:3]+positions[:,:,8:9])/2.

            # Put together new joint locations in SMPL skeleton structure
            self.num_joints = 22
            positions = torch.cat([old_pos[:,:,:6],
                                   spine2,
                                   old_pos[:,:,6:12],
                                   left_collar,
                                   right_collar,
                                   old_pos[:,:,12:]], dim=2)


        quats = self.inverse_kinematics(positions)
            
        if self.smpl_rotation_representation == 'rot_mat':
            rot = quaternion_to_matrix(quats)
        else:
            rot =  quaternion_to_axis_angle(quats)

        res = {}
        res['root_orient'] = rot[:,:,0]
        res['body_orient'] = rot[:,:,1:]
        res['trans'] = positions[:,:,0]

        return res
    
    def inverse_kinematics(self, joints):
        '''Get Forward Direction'''
        r_hip, l_hip, sdr_r, sdr_l = self.face_joint_idx
        across1 = joints[..., r_hip, :] - joints[..., l_hip, :]
        across2 = joints[..., sdr_r, :] - joints[..., sdr_l, :]
        across = across1 + across2
        across = across / torch.norm(across, dim=-1).unsqueeze(-1)
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        up = torch.zeros(joints.shape[:-2] + (3,)).to(joints.device)
        up[...,1] = 1.
        forward = torch.linalg.cross(up, across, axis=-1)
        forward = forward / torch.norm(forward, dim=-1).unsqueeze(-1)

        '''Get Root Rotation'''
        target = torch.zeros(joints.shape[:-2] + (3,)).to(joints.device)
        target[...,2] = 1.
        root_quat = quat_between_points(target, forward)


        # Initialize rotation parameters
        quat_params = torch.zeros(joints.shape[:-1] + (4,)).to(joints.device)
        quat_params[...,0] = 1.
        quat_params[:,:,0] = root_quat

        raw_offsets = self.get_buffer('smpl_raw_offsets')

        for jnt in list(self._children.keys()):
            if jnt == 0:
                continue
            u = torch.zeros(joints.shape[:-2] + (3,)).to(joints.device)
            v = torch.zeros(joints.shape[:-2] + (3,)).to(joints.device)
            # Rotations can only be determined for joints which have children
            if len(self._children[jnt]) > 0:

                # In case a joint has more than one child (root, spine3), average over children
                for chld in self._children[jnt]:
                    u += raw_offsets[chld]
                    v += (joints[..., chld, :] - joints[..., jnt, :]) / torch.sqrt(((joints[..., chld, :] - joints[..., jnt, :])**2).sum(axis=-1)).unsqueeze(-1)

                u = u / torch.norm(u, dim=-1).unsqueeze(-1)
                v = v / torch.norm(v, dim=-1).unsqueeze(-1)

                # We also need to consider the parents of a joint to negate their rotations
                prnt = self._parents[jnt]
                
                # To determine R, quat_params is recursively accessed. This increases the VRAM consumption during training.
                # Since we are only interested in the final R, it should be save to deactivate grads here
                with torch.no_grad():
                    # If no parents, no rotations need to be negated
                    if prnt == -1:
                        R = torch.zeros(joints.shape[:-2] + (4,)).to(joints.device)
                        R[:,0] = 1.
                    else:
                        R = quat_params[..., prnt, :]
                        # Travers all parents until root is found
                        while prnt != 0:
                            prnt = self._parents[prnt]
                            R_prev = quat_params[..., prnt, :]
                            R = multiply_quat(R_prev, R)

                # Get offsets without parent rotations
                v_rot = rotate_points_by_quaternion(invert_quat(R), v)

                # Determine rotation of joint as rotation between raw offset and offset of joints
                quat_params[...,jnt,:] = quat_between_points(u, v_rot)
                    
        return quat_params