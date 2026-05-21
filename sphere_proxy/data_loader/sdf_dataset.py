## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

## This source code is derived from DualSDF @ e43241c28e20eefb78ca7f467274bb56168dc1f0
##   (https://github.com/zekunhao1995/DualSDF)
## Copyright (c) 2020 Zekun Hao, licensed under the MIT license,
## cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os

import numpy as np
import torch
from torch.utils.data import Dataset

class SDFDataset(Dataset):
    def __init__(
        self,
        split_file,
        num_sdf_samples=2048,
        perc_sphere=0.1,
        perc_detail=None):
        """Dataset for the sphere proxy.
           Params:
                - data_root: Directory where the data is saved
                - num_sdf_samples: how many sdf samples should be loaded
                - perc_sphere: how many of these samples should be sphere samples. The rest 
                    is equally inside/outside the mesh"""
        with open(split_file, 'r') as f:
            self.filelist = f.readlines()

        # Remove new line characters from end of strings
        self.filelist = [f.replace("\n", "") for f in self.filelist]

        self.num_samples = num_sdf_samples
        self.perc_sphere = perc_sphere
        self.perc_detail = perc_detail

        tmp = np.load(self.filelist[0])
        self.smpl_shape_dim = tmp.shape[0]
        tmp = np.load(self.filelist[0].replace("joints", "shapes"))
        self.num_joints = tmp.shape[0]

        print(f"[SDFDataset] Loaded {self.__len__()} samples")



    def __len__(self):
        return len(self.filelist)


    def __getitem__(self, idx):
        joint_file = self.filelist[idx]
        shape_file = joint_file.replace("joints", "shapes")
        pose_file = joint_file.replace("joints", "poses")
        sphere_file = joint_file.replace("joints", "sphere")
        surface_file = joint_file.replace("joints", "surface")
        detail_file = joint_file.replace("joints", "detail")

        shape = np.load(shape_file)
        joints = np.load(joint_file)
        pose = np.load(pose_file)
        sphere = np.load(sphere_file, mmap_mode='r')
        surface = np.load(surface_file, mmap_mode='r')
        detail = np.load(detail_file, mmap_mode='r')

        if self.perc_detail is None:
            sample_perc_detail = detail.shape[0]/surface.shape[0]
        else:
            sample_perc_detail = self.perc_detail

        num_sphere_samples = int(self.num_samples*self.perc_sphere)
        num_detail_samples = int((self.num_samples - num_sphere_samples)*sample_perc_detail)

        # If not enough detail samples, fill missing samples with surface samples
        if num_detail_samples > detail.shape[0]:
            num_detail_samples = detail.shape[0]

        num_surface_samples = self.num_samples - num_sphere_samples - num_detail_samples
        num_inside_samples = num_surface_samples//2
        num_outside_samples = num_surface_samples - num_inside_samples

        # Take half of the surface samples inside the mesh and half outside the mesh
        # surface is [xyzd_inside, xyzd_outside] half inside, half outside
        # inside: d<0, outside: d>0
        tot_surface_samples = surface.shape[0]
        inside_idx = np.random.choice(tot_surface_samples//2, num_inside_samples, replace=False)
        outside_idx = np.random.choice(tot_surface_samples//2, num_outside_samples, replace=False) + tot_surface_samples//2
        surface_idx = np.concatenate([inside_idx, outside_idx])
        surface_samples = surface[surface_idx]

        # Sphere samples
        sphere_idx = np.random.choice(sphere.shape[0], num_sphere_samples, replace=False)
        sphere_samples = sphere[sphere_idx]

        # Detail samples
        detail_idx = np.random.choice(detail.shape[0], num_detail_samples, replace=False)
        detail_samples = detail[detail_idx]


        sdf_samples = np.concatenate([surface_samples, sphere_samples, detail_samples])
        # Joint ids in dataset are for SMPL-H, but we need joint ids for SMPL
        jnt_labels = conv_smplh_to_smpl(sdf_samples[:,4])

        data = {
            'shape': shape,
            'joints': joints,
            'pose': pose,
            'sdfs': sdf_samples[:,:4],
            'joint_labels': jnt_labels
        }

        return data

def conv_smplh_to_smpl(joint_ids):
    jnt_ids = np.copy(joint_ids).astype(int)
    l_mask = np.logical_and(jnt_ids > 21,jnt_ids <= 36)
    r_mask = jnt_ids > 36
    jnt_ids[l_mask] = 20
    jnt_ids[r_mask] = 21

    return jnt_ids
