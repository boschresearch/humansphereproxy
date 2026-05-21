## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0


import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--reg_dir", help="Directory of trained sphere proxy")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    config_path = os.path.join(args.reg_dir,"config_training.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_poses = np.load(os.path.join(args.reg_dir, "num_poses.npy"))
    coll_id_mat = np.load(os.path.join(args.reg_dir, "coll_id_mat.npy"))

    ### PLOT ###
    steps = np.linspace(0.02, 1.0, 50)
    plt_steps = np.linspace(0.02, 1.0, 50)
    sph_idx = np.arange(config['model']['num_spheres'])
    msk_tri = (sph_idx[:,None] < sph_idx)
    full_colls = len(np.where(msk_tri)[0])

    reduced_colls = []
    for i in steps:
        coll_split = (coll_id_mat >= i*num_poses)
        coll_red = np.logical_and(msk_tri, coll_split)
        reduced_colls.append(len(np.where(coll_red)[0]))
        if i == 0.9:
            print(f"Excluded sphere pairs: {len(np.where(coll_red)[0])/full_colls*100:.2f}%")#len(np.where(coll_red)[0])/full_colls*100)

    reduced_colls = [i*100/full_colls for i in reduced_colls]

    plt.plot(plt_steps*100, reduced_colls)
    plt.xlabel("Dataset percentage [%]")
    plt.ylabel("Sphere excluded [%]")
    plt.grid(axis='both', visible=True)
    plt.tight_layout()

    plt.savefig("SpherePairReduction.png")
    print()