## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
import smplx.lbs
import torch
import numpy as np
import smplx
import trimesh
import yaml

from sphere_proxy.models.joint_regressor import JointRegressor
from sphere_proxy.models.sphere_regressor import SphereRegressor

class SphereProxy(torch.nn.Module):
    """Approximation of a SMPL mesh using spheres."""
    def __init__(self,
            cpt_dir,
            collision_tolerance=0.0,
            collision_loss='L2',
            reduce_colls=True):
        
        super().__init__()

        config_pth = os.path.join(cpt_dir, "config_training.yaml")
        with open(config_pth, 'r') as f:
            self.config = yaml.safe_load(f)


        sphere_regressor_cpt = os.path.join(cpt_dir, "SR", "sphere_regressor.pth")
        joint_regressor_cpt = os.path.join(cpt_dir, "JR", "joint_regressor.pth")
        boneweight_cpt = os.path.join(cpt_dir, "boneweights.npy")
        coll_mat_cpt = os.path.join(cpt_dir, "coll_mat_1jd90.npy")

        self.num_spheres = int(self.config['model']['num_spheres'])
        self.latent_dim = int(self.config['model']['latent_dim'])
        self.num_joints = int(self.config['model']['num_joints'])
        self.smpl_shape_dim = int(self.config['model']['smpl_shape_dim'])

        # Load sphere regressor
        self.sr = SphereRegressor(self.num_spheres,
                                  self.smpl_shape_dim,
                                  self.latent_dim)
        self.sr.load_state_dict(torch.load(sphere_regressor_cpt, weights_only=True))
        self.sr.eval()

        # Load joint regressor
        self.jr = JointRegressor(self.smpl_shape_dim,
                                 self.num_joints,
                                 self.latent_dim)
        self.jr.load_state_dict(torch.load(joint_regressor_cpt, weights_only=True))
        self.jr.eval()

        # Load boneweights
        try:
            blend_weights = torch.from_numpy(np.load(boneweight_cpt))
            blend_weights.requires_grad = True
            self.register_buffer("blend_weights", blend_weights)
        except:
            self.register_buffer("blend_weights", torch.tensor([0.0]))

            print("No blendweights")
        
        # Set which spheres are allowed to collide with which spheres
        # Default: Only upper triangular matrix so no collisions are counted
        # twice
        sph_idx = torch.arange(self.num_spheres)
        coll_mask = (sph_idx[:,None] < sph_idx)
        self.reduce_colls = reduce_colls

        try:
            collision_indices = ~torch.from_numpy(np.load(coll_mat_cpt))
        except:
            collision_indices = None
            print("No collision indices")

        if collision_indices is not None and self.reduce_colls:
            coll_mask = torch.mul(coll_mask, collision_indices)           
        self.coll_idx = torch.where(coll_mask)

        # Set kinematic tree
        self.parents = np.array(self.config['mesh']['kin_tree'])   # (num_joints)

        self.centers = None # (bs, num_spheres, 3) sphere centers in rest pose
        self.radii = None # (bs, num_spheres) sphere centers in rest pose
        self.joints = None # (bs, num_joints, 3) joint locations in rest pose
        self.posed_centers = None # (bs, num_spheres, 3) sphere centers posed

        # Set loss parameters
        self.coll_tol = collision_tolerance
        self.coll_loss = collision_loss


    def forward(self, shape):
        """Set the parameters of the sphere proxy using the joint and sphere
            regressor."""
        sphere_params = self.sr(shape)
        self.centers = sphere_params[...,1:]
        self.radii = torch.exp(sphere_params[...,0])

        self.joints = self.jr(shape).reshape(-1, self.num_joints, 3)


    def selfintersection_loss(self):
        # If sphere proxy is posed, return posed loss, else return unposed loss
        if self.posed_centers is not None:
            cntrs = self.posed_centers
        else:
            cntrs = self.centers

        # Calculate distance between centers
        dists = cntrs[:,self.coll_idx[0][::]] - cntrs[:,self.coll_idx[1][::]]
        dists = torch.norm(dists, dim=-1)

        # Calculate allowed distance which is the sum of the radii
        dists_allow = self.radii[:,self.coll_idx[0][::]] + self.radii[:,self.coll_idx[1][::]] 

        # Loss is penetration depth...
        pen_dist = dists_allow - dists - self.coll_tol
        # ... but only for spheres which actually collide
        pen_dist[pen_dist<0.0] = 0.0

        if self.coll_loss == 'L1':
            loss = pen_dist
        elif self.coll_loss == 'L2':
            loss = pen_dist**2
        else:
            raise NotImplementedError

        # Sum over all spheres to get the loss for one pose
        return torch.sum(loss, dim=-1)



    def calc_sdf(self, points):
        """Evaluate the sdf of the sphere proxy at a given point.

            // The following fct is from DualSDF
            //   (https://github.com/zekunhao1995/DualSDF/blob/master/models/sdfsphere.py)
            // Copyright (c) 2020 Zekun Hao, licensed under the MIT license,
            // cf. 3rd-party-licenses.txt file in the root directory of this source tree.


            Params:
                points: points where to evaluate sdf (bs, num_points, 3)
        """
        if self.posed_centers is None:
            cntrs = self.centers.unsqueeze(-3)  # (bs, 1, num_spheres, 3)
        else:
            cntrs = self.posed_centers.unsqueeze(-3)
        radii = self.radii.unsqueeze(-2)    # (bs, 1, num_spheres)
        pnts = points.unsqueeze(-2)         # (bs, num_points, 1, 3)

        dists = torch.norm(pnts - cntrs, dim=-1) - radii # (bs, num_points, num_spheres)
        # TODO: Here we can obtain the sphere corresponding to the sdf value.
        # This can be used to apply the weighting for the arms and legs.
        sdf, _ = torch.min(dists, dim=-1)  # (bs, num_points)

        return sdf, dists
    
    def pose_spheres(self, pose, trans=None):
        bs = pose.shape[0]

        if self.joints.shape[0] == 1:
            self.joints = self.joints.expand([bs, -1, -1])

        if self.centers.shape[0] == 1:
            self.centers = self.centers.expand([bs, -1, -1])


        # Obtain the transformations between the joints
        _, trafos = smplx.lbs.batch_rigid_transform(pose, self.joints, self.parents)

        blend_weights = self.get_buffer("blend_weights")

        # Reshape so the dimensions can be casted togehter
        weighted_trafos = blend_weights @ trafos.reshape(bs, self.num_joints, 16)
        weighted_trafos = weighted_trafos.reshape(bs, self.num_spheres, 4, 4)

        # Convert spheres to homogeneous coordniates and apply transformations
        ones = torch.ones((bs, self.num_spheres, 1), device=self.centers.device)
        centers_homo = torch.concat([self.centers, ones], dim=-1)
        centers_homo_trans = weighted_trafos @ centers_homo.unsqueeze(-1)

        # Convert spheres back to cartesian coordinates
        # No need to divide by the homogeneous coordinate because the weights per sphere add to 1
        self.posed_centers = centers_homo_trans[:,:,:3,0]

        if trans is not None:
            self.posed_centers += trans

    def save_as_mesh(self, filename):
        sphere_meshes = []

        for i in range(self.num_spheres):
            tmp = trimesh.creation.icosphere(radius=self.radii[0,i].detach().cpu().numpy())
            if i == 0:
                tmp.visual.vertex_colors = [255,0,0,255]
            if i == 50:
                tmp.visual.vertex_colors = [0,255,0,255]
            if i == 100:
                tmp.visual.vertex_colors = [0,0,255,255]
            if self.posed_centers is None:
                tmp.apply_translation(self.centers[0,i].detach().cpu().numpy())
            else:
                tmp.apply_translation(self.posed_centers[0,i].detach().cpu().numpy())

            sphere_meshes.append(tmp)
        
        sphere_proxy = trimesh.util.concatenate(sphere_meshes)
        sphere_proxy.export(filename)


    def visualize_collision_matrix(self, filename, coll_mat, id=0):
        sphere_meshes = []

        for i in range(self.num_spheres):
            tmp = trimesh.creation.icosphere(radius=self.radii[0,i].detach().cpu().numpy())
            if i == id:
                tmp.visual.vertex_colors = [0,255,0,255]
            if coll_mat[id,i] or coll_mat[i,id]:
                tmp.visual.vertex_colors = [255,0,0,255]

            if self.posed_centers is None:
                tmp.apply_translation(self.centers[0,i].detach().cpu().numpy())
            else:
                tmp.apply_translation(self.posed_centers[0,i].detach().cpu().numpy())

            sphere_meshes.append(tmp)
        
        sphere_proxy = trimesh.util.concatenate(sphere_meshes)
        sphere_proxy.export(filename)       



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import smplx
    from models.sphere_regressor import SphereDecoder
    from humanml3d_to_smpl.utils.params import beta_hml3d
    import os
    import yaml
    smplh_path = "../mdm_fork/body_models/smplh_merged/SMPLH_NEUTRAL.pkl"
    smpl_layer = smplx.SMPLHLayer(smplh_path).to(device)
    pose = torch.from_numpy(np.load("data/sdfs_smpl_posed/0_poses/00000000.npy")).unsqueeze(0).to(device)
    kin_tree = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    coll_mat = np.load("save/2024-May-31-14-26-45/coll_mat_4jd90.npy")
    shape = torch.tensor([beta_hml3d]).to(device)


    for sh_id in range(10):
        #shape = ((torch.rand(10) - 0.5)*10).unsqueeze(0).to(device)
        #print("Shape")
        #print(shape)
        #print("\n")

        #shape = torch.Tensor([pert]).to(device)
        mesh = smpl_layer(betas=shape)


        d_verts = mesh.vertices
        d_verts = d_verts[:, smpl_layer.faces.astype(np.int32)]
        d_verts = d_verts.reshape(1,-1,3).detach().cpu().numpy()

        # Center mesh around zero
        mesh_max = np.amax(d_verts, axis=1)
        mesh_min = np.amin(d_verts, axis=1)
        mesh_center = (mesh_max + mesh_min) / 2
        d_verts = d_verts - mesh_center[:,np.newaxis,...]

        # Scale mesh to be in [-1,1]
        max_dist = np.sqrt(np.max(np.sum(d_verts**2, axis=-1), axis=1))
        mesh_scale = 1.0 / max_dist

        mesh_center = torch.from_numpy(mesh_center).to(device)
        mesh_scale = torch.from_numpy(mesh_scale).to(device)
        rest_joints = mesh.joints[:,:22]
        rest_joints -= mesh_center
        rest_joints *= mesh_scale

        bw = torch.ones((1, 2, len(kin_tree)), device=device)/22.0

        checkpoint_dir = "./save/2024-May-31-14-26-45"
        run = checkpoint_dir.split("/")[2]
        with open(os.path.join(checkpoint_dir, "config_training_sphere_proxy.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        sp_reg = SphereDecoder(config['model']['num_spheres'], 10, config['model']['latent_dim']).to(device)
        sp_reg.load_state_dict(torch.load(os.path.join(checkpoint_dir, "sphereRegressor.pth")))
        sp_reg.eval()

        sphere_params = sp_reg(shape)

        rads = torch.sort(torch.exp(sphere_params[:,:,0]))
        min_rad = torch.min(torch.exp(sphere_params[:,:,0]))
        max_rad = torch.max(torch.exp(sphere_params[:,:,0]))
        print("Shape 0")
        print(f"Min rad: {min_rad}")
        print(f"Max rad: {max_rad}")

        # shape = torch.Tensor([[-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.]]).to('cuda')
        # sphere_params = sp_reg(shape)

        # min_rad = torch.min(torch.exp(sphere_params[:,:,0]))
        # max_rad = torch.max(torch.exp(sphere_params[:,:,0]))
        # print("Shape -5")
        # print(f"Min rad: {min_rad}")
        # print(f"Max rad: {max_rad}")

        # shape = torch.Tensor([[5.,5.,5.,5.,5.,5.,5.,5.,5.,5.]]).to('cuda')
        # sphere_params = sp_reg(shape)

        # min_rad = torch.min(torch.exp(sphere_params[:,:,0]))
        # max_rad = torch.max(torch.exp(sphere_params[:,:,0]))
        # print("Shape 5")
        # print(f"Min rad: {min_rad}")
        # print(f"Max rad: {max_rad}")

        sp = SphereProxy(sphere_params[:,:,1:], torch.exp(sphere_params[:,:,0]), rest_joints, kin_tree, bw)
        sp.visualize_collision_matrix("coll_mat_test.obj", coll_mat, 0)
        exit()

        centers = sphere_params[:,:,1:].unsqueeze(2)
        joints = rest_joints.unsqueeze(1)
        diff = centers - joints
        diff = torch.norm(diff, dim=-1)     # (bs, num_spheres, num_joints)

        assign = torch.argmin(diff, dim=-1)[0].cpu().numpy()

        cnts = {}
        for i in assign:
            if i not in cnts:
                cnts[i] = 1
            else:
                cnts[i] += 1
        print(dict(sorted(cnts.items())))

        sp.save_as_mesh(os.path.join(checkpoint_dir,f"sphereTest_{sh_id}.obj"))
        # mesh_tri = trimesh.Trimesh(vertices=mesh.vertices[0].detach().cpu().numpy(), faces=smpl_layer.faces)
        # mesh_tri.export(os.path.join(checkpoint_dir,"gt.stl"))
        print()