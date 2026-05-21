## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import torch

def clamped_outside_SDF_loss(pred, gt, weight):
    """Calculate difference between pred and gt for outside sdf values.
       Weight can be used to give different importance to different spheres, i.e.,
       give more importance to arms and legs"""
    in_mask = gt < 0.0
    out_mask = torch.logical_not(in_mask)
    loss = torch.empty_like(gt)
    loss[out_mask] = torch.abs(gt[out_mask] - pred[out_mask])
    loss[in_mask] = torch.clamp(pred[in_mask], min=0.0)
    loss = torch.mean(loss *  weight)

    return loss


def emptiness_loss(sphere_point_dists, gt_sdfs, max_tol_dist=0.05):
   """Penalize spheres that are located outside the body (no points with negative sdf inside the sphere)
      Params:
         sphere_point_dists: (bs, num_points, num_spheres)
         gt_sdfs: (bs, num_points)
      Returns
         loss: (bs) >= 0"""
   # We are only interested in inside points, i.e., points with sdf < 0
   in_mask = gt_sdfs < 0
   dist_to_inside_points = torch.ones_like(sphere_point_dists)
   dist_to_inside_points[in_mask,:] = sphere_point_dists[in_mask,:]
   min_dists, _ = torch.min(dist_to_inside_points, dim=1)

   # Add a tolerance to allow for small violations of the inside point assumption
   min_dists_tol = min_dists - max_tol_dist
   clamped_dist = torch.clamp(min_dists_tol, min = 0.0)
   loss = torch.mean(clamped_dist)

   return loss


def boneweight_regularization(bone_weight_matrix):
   abs_weights = torch.abs(bone_weight_matrix)
   abs_sum = torch.sum(abs_weights, dim=-1)
   loss = torch.mean(abs_sum)

   return loss


def selfintersection_loss(centers, radii):
    radii_1 = radii.unsqueeze(-1)
    radii_2 = radii.unsqueeze(-2)
    cntrs_1 = centers.unsqueeze(-2)
    cntrs_2 = centers.unsqueeze(-3)

    diff = cntrs_1 - cntrs_2
    dis = torch.norm(diff, dim=-1)
    dis_all = radii_1 + radii_2
    msk_dists = (dis >= dis_all)
    loss = dis_all - dis
    # # Set diagonal to zero as these spheres will always collide
    loss.diagonal(dim1=-1, dim2=-2).zero_()
    # # Matrices are symmetric, we only need one half
    loss = torch.triu(loss)
    # # Remove entries of spheres which do not collide
    loss[msk_dists] = 0.
    loss = torch.mean(loss)

    return loss


def joint_distribution_loss(centers, joints, weights):
   num_joints = torch.Tensor([joints.shape[1]])
   smn = torch.nn.Softmin(dim=-1)
   smx = torch.nn.Softmax(dim=-1)
   centers = centers.unsqueeze(2)
   joints = joints.unsqueeze(1)
   diff = centers - joints
   diff = torch.norm(diff, dim=-1)     # (bs, num_spheres, num_joints)

   # min_v, min_i = torch.min(diff, dim=-1)

   # assign = torch.zeros((joints.shape[0], joints.shape[2])).to(joints.device)

   # for i in range(joints.shape[2]):
   #    msk = min_i == i
   #    sm = torch.sum(msk, dim=1)
   #    assign[:, i] = sm

   assign = smn(diff)
   assign = torch.sum(assign, dim=1)   # (bs, num_joints)
   
   weights = weights.unsqueeze(0)
   assign /= weights

   dist = smx(assign)
   log_dist = torch.log(dist)


   # Maximize entropy, i.e., get equal distribution of spheres
   entropy = -1 * dist * log_dist

   #entropy *= weights
   entropy = torch.sum(entropy, dim=-1)

   # Objectives are minimized so we need to invert it
   entropy *= -1

   # Add max value so lowest possible value is 0
   log_joints = torch.log(num_joints).to(entropy.device)
   entropy += log_joints

   loss = torch.mean(entropy)

   return loss


def radius_loss(radii, min_rad):
   msk = radii < min_rad
   loss = torch.zeros_like(radii)
   loss[msk] = min_rad - radii[msk]
   loss = torch.mean(loss)

   return loss
