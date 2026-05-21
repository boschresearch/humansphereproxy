## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import numpy as np
from PIL import Image
import os
import cv2
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--base_dir")
    parser.add_argument("--out_name")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    num_files = len(os.listdir(os.path.join(args.base_dir, "tmp")))
    title = np.array(Image.open(os.path.join(args.base_dir, "Title.png")))

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(os.path.join(args.base_dir, args.out_name), fourcc=fourcc, fps=20, frameSize=(1920,1080))

    for i in range(300):
        out.write(title)


    # MoMask HumanML3D
    # for i in range(num_files):
    #     if i > 3327:
    #         continue
    #     if i >= 768 and i <= 1023:
    #         continue
    #     if i >= 1280 and i <= 1535:
    #         continue
    #     if i >= 1536 and i <= 1791:
    #         continue
    #     if i >= 2304 and i <= 2559:
    #         continue
    #     if i >= 2816 and i <= 3071:
    #         continue
    #     if i >= 4352 and i <= 4607:
    #         continue
    #     if i >= 4864 and i <= 5119:
    #         continue
    #     if i >= 6144 and i <= 6399:
    #         continue
    #     if i >= 7168 and i <= 7423:
    #         continue
    #     fn = os.path.join(args.base_dir, "tmp", f"{i}.png")
    #     frm = cv2.cvtColor(np.array(Image.open(fn)),cv2.COLOR_BGR2RGB)
    #     out.write(frm)


    #MoMask KIT
    # for i in range(num_files):
    #     if i > 1535:
    #         continue

    #     fn = os.path.join(args.base_dir, "tmp", f"{i}.png")
    #     frm = cv2.cvtColor(np.array(Image.open(fn)),cv2.COLOR_BGR2RGB)
    #     out.write(frm)


    # MDM HumanML3D
    for i in range(num_files):
        if i > 2544:
            continue

        fn = os.path.join(args.base_dir, "tmp", f"{i}.png")
        frm = cv2.cvtColor(np.array(Image.open(fn)),cv2.COLOR_BGR2RGB)
        out.write(frm)


