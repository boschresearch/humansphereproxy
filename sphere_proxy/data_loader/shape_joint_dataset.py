## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import numpy as np
from torch.utils.data import Dataset

class ShapeJointDataset(Dataset):
    def __init__(self, split_file, data_transform=None, target_transform=None):
        with open(split_file, 'r') as f:
            self.filelist = f.readlines()

        # Remove new line characters from end of strings
        self.filelist = [f.replace("\n", "") for f in self.filelist]

        tmp_joints = np.load(self.filelist[0])
        tmp_shape = np.load(self.filelist[0].replace("joints", "shapes"))
        self.shape_dim = tmp_shape.shape[0]
        self.num_joints = tmp_joints.shape[0]

        self.data_transform = data_transform
        self.target_transform = target_transform

        print(f"[ShapeJointDataset] Loaded {self.__len__()} samples")


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        joint_path = self.filelist[idx]
        shape_path = joint_path.replace("joints", "shapes")

        shape = np.load(shape_path)
        joints = np.load(joint_path)

        if self.data_transform:
            shape = self.data_transform(shape)

        if self.target_transform:
            joints = self.target_transform(joints) 

        return shape, joints
