## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch


def quat_between_points_np(x,y):
    # https://math.stackexchange.com/questions/4520571/how-to-get-a-rotation-quaternion-from-two-vectors
    n = np.cross(x, y)
    c = np.sum(x * y, axis=-1)/(np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1))

    w = np.sqrt((1+c)/(1-c))*np.linalg.norm(n, axis=-1)

    quat = np.concatenate([w[..., np.newaxis], n], axis=-1)

    return quat/np.linalg.norm(quat, axis=-1, keepdims=True)


def rotate_points_by_quaternion_np(q, v):
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#:~:text=Quaternion%2Dderived%20rotation%20matrix
    v_outer = np.expand_dims(q[...,1:], axis=-1) @ np.expand_dims(q[...,1:], axis=-2)
    v_cross = np.cross(q[...,1:], v)

    part1 = np.squeeze(v_outer @ np.expand_dims(v, axis=-1), axis=-1)
    part2 = (q[...,0:1]**2)*v
    part3 = 2*q[...,0:1]*v_cross
    part4 = np.cross(q[...,1:], v_cross)

    return part1 + part2 + part3 + part4


def multiply_quat_np(q, p):
    # https://en.wikipedia.org/wiki/Quaternion#:~:text=then%20the%20formulas%20for%20addition%2C%20multiplication%2C%20and%20multiplicative%20inverse%20are
    v_dot = np.squeeze(np.expand_dims(q[...,1:], axis=-2) @ np.expand_dims(p[...,1:], axis=-1),axis=-1)
    v_cross = np.cross(q[...,1:], p[...,1:])

    w = q[...,:1]*p[...,:1] - v_dot
    v = q[...,:1]*p[...,1:] + p[...,:1]*q[...,1:] + v_cross

    return np.concatenate([w,v], axis=-1)


def invert_quat(q):
    # https://en.wikipedia.org/wiki/Quaternion#:~:text=.%20The%20conjugate%20of%20q%20is%20the%20quaternion
    inv_quat = q.clone()
    inv_quat[...,1:] *= -1
    return inv_quat



def multiply_quat(q, p):
    # https://en.wikipedia.org/wiki/Quaternion#:~:text=then%20the%20formulas%20for%20addition%2C%20multiplication%2C%20and%20multiplicative%20inverse%20are
    v_dot = (q[...,1:].unsqueeze(-2) @ p[...,1:].unsqueeze(-1)).squeeze(-1)
    v_cross = torch.linalg.cross(q[...,1:], p[...,1:])

    w = q[...,:1]*p[...,:1] - v_dot
    v = q[...,:1]*p[...,1:] + p[...,:1]*q[...,1:] + v_cross

    return torch.concatenate([w,v], axis=-1)


def quat_between_points(x,y):
    # https://math.stackexchange.com/questions/4520571/how-to-get-a-rotation-quaternion-from-two-vectors
    n = torch.linalg.cross(x, y)
    c = torch.sum(x * y, axis=-1)/(torch.linalg.norm(x, axis=-1)*torch.linalg.norm(y, axis=-1))

    w = torch.sqrt((1+c)/(1-c))*torch.linalg.norm(n, axis=-1)

    quat = torch.concatenate([w[..., np.newaxis], n], axis=-1)

    return quat/torch.linalg.norm(quat, axis=-1, keepdims=True)


def rotate_points_by_quaternion(q, v):
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#:~:text=Quaternion%2Dderived%20rotation%20matrix
    v_outer = q[...,1:].unsqueeze(-1) @ q[...,1:].unsqueeze(-2)
    v_cross = torch.linalg.cross(q[...,1:], v)

    part1 = (v_outer @ v.unsqueeze(-1)).squeeze(-1)
    part2 = (q[...,0:1]**2)*v
    part3 = 2*q[...,0:1]*v_cross
    part4 = torch.linalg.cross(q[...,1:], v_cross)

    return part1 + part2 + part3 + part4

