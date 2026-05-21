## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--split", nargs=3, type=float, default=[0.8,0.15,0.05], help="Set split into train, test, and val")
    parser.add_argument("--data_pth", help="Path to data", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    print(args.split)

    assert sum(args.split) == 1.0, "Split must sum to 1"

    train_files = []
    test_files = []
    val_files = []

    loc = os.path.join(args.data_pth, "0_joints")

    curr_files = os.listdir(loc)
    num_files = len(curr_files)
    cnt = 0
    for curr_file in curr_files:
        curr_file += "\n"
        if cnt < int(args.split[0]*num_files):
            train_files.append(os.path.join(loc, curr_file))
        elif cnt < int((args.split[0]+args.split[1])*num_files):
            test_files.append(os.path.join(loc, curr_file))
        else:
            val_files.append(os.path.join(loc, curr_file))
        cnt += 1

    with open(os.path.join(args.data_pth, "train.txt"), 'a') as f:
        f.writelines(train_files)

    with open(os.path.join(args.data_pth, "test.txt"), 'a') as f:
        f.writelines(test_files)

    with open(os.path.join(args.data_pth, "val.txt"), 'a') as f:
        f.writelines(val_files)

