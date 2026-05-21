## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os

import torch
from torch.utils.data import DataLoader

from sphere_proxy.models.joint_regressor import JointRegressor
from sphere_proxy.data_loader.shape_joint_dataset import ShapeJointDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def calc_metric(pred, label):
    diff = pred - label
    return torch.norm(diff, dim=-1)


if __name__ == "__main__":
    model_dir = "save/SP-128/JR"

    ### Load data
    print("Loading data...")
    data_path = "dataset/sdfs_smpl_unposed/test.txt"
    dataset = ShapeJointDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


    ### Load model
    print("Loading model...")
    checkpoint = "joint_regressor.pth"
    model = JointRegressor(dataset.shape_dim, dataset.num_joints, latent_dim=512)
    model.load_state_dict(torch.load(os.path.join(model_dir,checkpoint)))
    model.to(device)
    model.eval()

    joint_dists = None

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            pred = pred.reshape(data.shape[0], dataloader.dataset.num_joints, 3)

            # Denormalize labels and predictions to calculate metric in meters
            #pred = joint_denormalizer(pred)
            #label = joint_denormalizer(label)


            if joint_dists is not None:
                joint_dists_batch = calc_metric(pred, label)
                joint_dists = torch.cat((joint_dists, joint_dists_batch))
            else:
                joint_dists = calc_metric(pred, label)

    mean_dist = torch.mean(joint_dists, dim=0).detach().cpu().numpy()
    std_dist = torch.std(joint_dists, dim=0).detach().cpu().numpy()

    max_dist = torch.max(joint_dists).detach().cpu().numpy()
    min_dist = torch.min(joint_dists).detach().cpu().numpy()

    mean_all = torch.mean(joint_dists).detach().cpu().numpy()
    std_all = torch.std(joint_dists).detach().cpu().numpy()

    stat_str = ""
    for i in range(len(mean_dist)):
        stat_str += f"Mean dist for joint {i}: {100*mean_dist[i]:0.2f} +/- {100*std_dist[i]:0.2f} [cm]\n"

    stat_str += f"\nHighest dist: {100*max_dist:0.2f} [cm]\n"
    stat_str += f"Lowest dist: {100*min_dist:0.2f} [cm]\n"
    stat_str += f"Overall mean dist: {100*mean_all:0.2f} +/- {100*std_all:0.2f} [cm]"

    print(stat_str)
    with open(os.path.join(model_dir, "test_stats.txt"), "w") as eval_file:
        eval_file.write(stat_str)
