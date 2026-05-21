## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

## This source code is derived from DualSDF @ e43241c28e20eefb78ca7f467274bb56168dc1f0
##   (https://github.com/zekunhao1995/DualSDF)
## Copyright (c) 2020 Zekun Hao, licensed under the MIT license,
## cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import os
import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import smplx

from extensions.mesh2sdf2_cuda import mesh2sdf
from util.pcl_library import mesh2pcl

def conv_smplh_to_smpl(joint_id):
    if joint_id <= 21:
        return joint_id
    elif joint_id <= 36:
        return 20
    else:
        return 21


def determine_bone_weights(smpl_lbs_weights, verts, points):
    minibatches = 25
    num_points = len(points)
    bs = num_points//minibatches
    verts = verts.unsqueeze(0)
    points = points.unsqueeze(1)

    bone_weights = torch.Tensor(num_points).fill_(-1).to(verts.device)


    for mb in range(minibatches):
        pnts = points[mb*bs:(mb+1)*bs]
        dists = torch.norm(verts-pnts, dim=-1)
        min_dists, min_indx = torch.min(dists, dim=1)#, k=4, dim=1, largest=False)
        weights = smpl_lbs_weights[min_indx.detach().cpu()]
        weights_val, weights_idx = torch.max(weights, dim=-1)
        bone_weights[mb*bs:(mb+1)*bs] = weights_idx
        # for i in range(len(weights_idx)):
        #     bone_weights[mb*bs+i] = conv_smplh_to_smpl(weights_idx[i])
    
    return bone_weights

def sdfmeshfun(point, mesh):
    out_ker = mesh2sdf.mesh2sdf_gpu(point.contiguous(),mesh)[0]
    return out_ker
    
    
def meshpreprocess_bsphere(mesh_path, faces):
    joint_path = mesh_path.replace("triangles", "joints")
    shape_path = mesh_path.replace("triangles", "shapes")
    pose_path = mesh_path.replace("triangles", "poses")

    verts = np.load(mesh_path)
    joints = np.load(joint_path)
    shape = np.load(shape_path)
    pose =  np.load(pose_path)
    mesh = verts[faces.astype(np.int32)]

    # Mirror y-coordinate
    #mesh[:,:,1] *= -1
    #joints[:,1] *= -1

    '''Meshes are already normalized in preprocessing'''

    # normalize mesh
    # mesh = mesh.reshape(-1,3)
    # mesh_max = np.amax(mesh, axis=0)
    # mesh_min = np.amin(mesh, axis=0)
    # mesh_center = (mesh_max + mesh_min) / 2
    # mesh = mesh - mesh_center
    # # Apply translation also to joints
    # joints = joints - mesh_center
    # # Find the max distance to origin
    # max_dist = np.sqrt(np.max(np.sum(mesh**2, axis=-1)))
    # mesh_scale = 1.0 / max_dist
    # mesh *= mesh_scale
    # # Apply scaling also to joints
    # joints *= mesh_scale

    # # Plot test to check new joint locations
    # # import matplotlib.pyplot as plt

    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.set_xlim(-1,1)
    # # ax.set_ylim(-1,1)
    # # ax.set_zlim(-1,1)
    # # ax.scatter(mesh[::5,0], mesh[::5,1], mesh[::5,2], c='green', alpha=0.1)
    # # ax.scatter(joints[:,0], joints[:,1], joints[:,2], c='red')
    # # plt.savefig("mesh_joint_test.png")

    # mesh = mesh.reshape(-1,3,3)
    mesh_t = torch.from_numpy(mesh.astype(np.float32)).contiguous()
    verts_t = torch.from_numpy(verts.astype(np.float32)).contiguous()
    return mesh_t, joints, pose, shape, verts_t

def normalize(x):
    x /= torch.sqrt(torch.sum(x**2))
    return x

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = args.mesh_npy_path

    data_list = []
    with os.scandir(data_path) as npy_list:
        for npy_path in npy_list:
            if npy_path.is_file():
                data_list.append(npy_path.path)
    data_list.sort()
    print(len(data_list))
    num_shapes = len(data_list)

    target_path = args.output_path
    
    # for each mesh, sample points within bounding sphere.
    # According to DeepSDF paper, 250,000x2 points around the surface,
    # 25,000 points within the unit sphere uniformly
    # To sample points around the surface, 
    #   - sample points uniformly on the surface,
    #   - Perturb the points with gaussian noise var=0.0025 and 0.00025
    #   - Then compute SDF
    num_surface_samples = 320000
    num_sphere_samples = 250000
    target_samples = 250000

    noise_vec = torch.empty([num_surface_samples,3], dtype=torch.float32, device=device) # x y z
    noise_vec2 = torch.empty([num_sphere_samples,3], dtype=torch.float32, device=device) # x y z
    noise_vec3 = torch.empty([num_sphere_samples,1], dtype=torch.float32, device=device) # x y z

    smplLayer = smplx.SMPLHLayer("body_models/smplh_merged/SMPLH_NEUTRAL.pkl")
    smpl_weights = smplLayer.lbs_weights
    faces = smplLayer.faces.astype(np.int32)


    for shape_id in range(args.resume, len(data_list)):
        print('Processing {} - '.format(shape_id), end='')
        mesh_path = data_list[shape_id]
        mesh_path_split = mesh_path.split('/')
        classid = mesh_path_split[-2]
        shapeid = mesh_path_split[-1].split('.')[0]
        print(classid, shapeid)
        
        start = time.time()
        
        mesh, joints, pose, shape, verts = meshpreprocess_bsphere(mesh_path, faces)
        mesh = mesh.to(device)
        verts = verts.to(device)

        pcl = torch.from_numpy(mesh2pcl(mesh.cpu().numpy(), num_surface_samples)).to(device) # [N, 3]
        
        # Surface points
        noise_vec.normal_(0, np.sqrt(0.005))
        points1 = pcl + noise_vec
        noise_vec.normal_(0, np.sqrt(0.0005))
        points2 = pcl + noise_vec
        
        # Unit sphere points
        noise_vec2.normal_(0, 1)
        shell_points = noise_vec2 / torch.sqrt(torch.sum(noise_vec2**2, dim=-1, keepdim=True))
        noise_vec3.uniform_(0, 1) # r = 1
        points3 = shell_points * (noise_vec3**(1/3))

        all_points = torch.cat([points1, points2, points3], dim=0)

        point_joint_labels = determine_bone_weights(smpl_weights, verts, all_points)
        
        
        #print(all_points.shape)
        sample_dist = sdfmeshfun(all_points, mesh)
        #print(sample_dist.shape)
        
        xyzd = torch.cat([all_points, sample_dist.unsqueeze(-1), point_joint_labels.unsqueeze(-1)], dim=-1).cpu().numpy()
        
        xyzd_sur = xyzd[:num_surface_samples*2]
        xyzd_sph = xyzd[num_surface_samples*2:]
        
        inside_mask = (xyzd_sur[:,3] <= 0)
        outside_mask = np.logical_not(inside_mask)

        inside_cnt = np.count_nonzero(inside_mask)
        outside_cnt = np.count_nonzero(outside_mask)
        inside_stor = [xyzd_sur[inside_mask,:]]
        outside_stor = [xyzd_sur[outside_mask,:]]
        n_attempts = 0
        badsample = False
        while (inside_cnt < target_samples) or (outside_cnt < target_samples):
            noise_vec.normal_(0, np.sqrt(0.005))
            points1 = pcl + noise_vec
            noise_vec.normal_(0, np.sqrt(0.0005))
            points2 = pcl + noise_vec
            all_points = torch.cat([points1, points2], dim=0)
            point_joint_labels = determine_bone_weights(smpl_weights, verts, all_points)
            sample_dist = sdfmeshfun(all_points, mesh)
            xyzd_sur = torch.cat([all_points, sample_dist.unsqueeze(-1), point_joint_labels.unsqueeze(-1)], dim=-1).cpu().numpy()
            inside_mask = (xyzd_sur[:,3] <= 0)
            outside_mask = np.logical_not(inside_mask)
            inside_cnt += np.count_nonzero(inside_mask)
            outside_cnt += np.count_nonzero(outside_mask)
            inside_stor.append(xyzd_sur[inside_mask,:])
            outside_stor.append(xyzd_sur[outside_mask,:])
            n_attempts += 1
            print(" - {}nd Attempt: {} / {}".format(n_attempts, inside_cnt, target_samples))
            if n_attempts > 200 or ((np.minimum(inside_cnt, outside_cnt)/n_attempts) < 500):
                with open('bads_list_{}.txt'.format(classid), 'a+') as f:
                    f.write('{},{},{},{}\n'.format(classid, shapeid, np.minimum(inside_cnt, outside_cnt), n_attempts))
                badsample = True
                break
            
        xyzd_inside = np.concatenate(inside_stor, axis=0)
        xyzd_outside = np.concatenate(outside_stor, axis=0)
        
        num_yields = np.minimum(xyzd_inside.shape[0], xyzd_outside.shape[0])
        xyzd_inside = xyzd_inside[:num_yields,:]
        xyzd_outside = xyzd_outside[:num_yields,:]
        
        xyzd = np.concatenate([xyzd_inside, xyzd_outside], axis=0)

        detail_la = xyzd[:,4] == 7               # left ankle
        detail_ra = xyzd[:,4] == 8               # right ankle
        detail_lf = xyzd[:,4] == 10              # left_foot
        detail_rf = xyzd[:,4] == 11              # right_foot
        detail_h = xyzd[:,4] >= 20               # hands
        detail = np.logical_or(detail_lf, detail_rf)
        detail = np.logical_or(detail, detail_h)
        detail = np.logical_or(detail, detail_la)
        detail = np.logical_or(detail, detail_ra)

        xyzd_sur = xyzd[np.where(~detail)]
        xyzd_det = xyzd[np.where(detail)]
        
        end = time.time()
        print("[Perf] time: {}, yield: {}".format(end - start, num_yields))
        
        save_path = os.path.join(target_path, classid+"_surface")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,'{}.npy'.format(shapeid)), xyzd_sur)

        save_path = os.path.join(target_path, classid+"_detail")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,'{}.npy'.format(shapeid)), xyzd_det)

        save_path = os.path.join(target_path, classid+"_sphere")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,'{}.npy'.format(shapeid)), xyzd_sph)

        save_path = os.path.join(target_path, classid+"_joints")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, '{}.npy'.format(shapeid)), joints)

        save_path = os.path.join(target_path, classid+"_shapes")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, '{}.npy'.format(shapeid)), shape)

        save_path = os.path.join(target_path, classid+"_poses")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, '{}.npy'.format(shapeid)), pose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample SDF values from meshes. All the NPY files under mesh_npy_path and its child dirs will be converted and the directory structure will be preserved.')
    parser.add_argument('mesh_npy_path', type=str,
                        help='The dir containing meshes in NPY format [ #triangles x 3(vertices) x 3(xyz) ]')
    parser.add_argument('output_path', type=str,
                        help='The output dir containing sampled SDF in NPY format [ #points x 4(xyzd) ]')
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()
    main(args)
    
    
    
    
