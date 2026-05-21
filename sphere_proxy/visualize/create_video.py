## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from argparse import ArgumentParser
import numpy as np
import torch
from humanml3d_to_smpl.humanml3d_to_smpl_layer import HumanML3D_To_SMPL_Layer
from humanml3d_to_smpl.utils import params
import smplx
import trimesh
import yaml
import pyrender
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from extensions.mesh2sdf2_cuda import mesh2sdf
from util.util import get_mesh_normalization


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--motion_path_baseline", type=str)
    parser.add_argument("--motion_path_sia", type=str)
    parser.add_argument("--baseline_name", type=str)
    parser.add_argument("--sia_name", type=str)
    parser.add_argument("--out_dir", default="./visualize_objs", type=str)
    parser.add_argument("--si_prec", default=0.02, type=float, help="Set negative to disable si calculation")
    parser.add_argument("--dataset", choices=['kit', 'hml'])

    return parser.parse_args()


def calculate_selfintersection_volume(vertices, faces, prec=0.06):
    incounts = mesh2sdf.calc_incount(vertices[:,faces], prec)
    vols = (torch.sum(incounts, dim=1)*prec**3)*1e6
    incounts[incounts<2] = 0
    vols_is = (torch.sum(incounts, dim=1)*prec**3)*1e6
    vols = vols.detach().cpu().numpy()
    vols_is = vols_is.detach().cpu().numpy()
    return vols_is, 100*vols_is/vols

def Rz(angle):
    rad = angle*2*np.pi/360

    return np.array([[np.cos(rad), -np.sin(rad), 0.0],
                     [np.sin(rad), np.cos(rad), 0.0],
                     [0.0, 0.0, 1.0]])

def Ry(angle):
    rad = angle*2*np.pi/360

    return np.array([[np.cos(rad), 0.0, np.sin(rad)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(rad), 0.0, np.cos(rad)]])

def Rx(angle):
    rad = angle*2*np.pi/360

    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(rad), -np.sin(rad)],
                     [0.0, np.sin(rad), np.cos(rad)]])

def setup_scene(motion_extends):
    dims_max = np.max(motion_extends, axis=0)
    dims_min = np.min(motion_extends, axis=0)

    dims_cntr = (dims_max+dims_min)/2.
    dims = dims_max-dims_min

    scene_all = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
    scene_sis = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    light1 = pyrender.PointLight(color=[1.0,1.0,1.0], intensity=2.0)
    light2 = pyrender.PointLight(color=[1.0,1.0,1.0], intensity=2.0)
    light3 = pyrender.PointLight(color=[1.0,1.0,1.0], intensity=2.0)
    light4 = pyrender.PointLight(color=[1.0,1.0,1.0], intensity=2.0)

    #Floor settings
    floor = trimesh.creation.box(extents=[dims[0], dims[1], 0.05])
    floor.unmerge_vertices()
    floor.visual.face_colors = [155, 155, 155, 255]
    floor_pr = pyrender.Mesh.from_trimesh(floor, smooth=False)
    floor_pose = np.eye(4)
    floor_pose[:3,3] = [dims_cntr[0], dims_cntr[1], dims_min[2]-0.025]
    scene_all.add(floor_pr, name="floor", pose=floor_pose)

    # Light settings
    light1_pose = np.eye(4)
    light1_pose[:3,3] = [dims_max[0],dims_max[1],dims_max[2]+2.0]
    scene_all.add(light1, name="light1", pose=light1_pose)

    light2_pose = np.eye(4)
    light2_pose[:3,3] = [dims_min[0],dims_max[1],dims_max[2]+2.0]
    scene_all.add(light2, name="light2", pose=light2_pose)

    light3_pose = np.eye(4)
    light3_pose[:3,3] = [dims_max[0],dims_min[1],dims_max[2]+2.0]
    scene_all.add(light3, name="light3", pose=light3_pose)

    light4_pose = np.eye(4)
    light4_pose[:3,3] = [dims_min[0],dims_min[1],dims_max[2]+2.0]
    scene_all.add(light4, name="light4", pose=light4_pose)

    return scene_all, scene_sis, camera


def render_smpl_motion(smpl_shape, root_orient, body_pose, translations, cam_settings, calc_si_prec=0.02):
    """
        Given mesh parameters of a motion and a camera pose, render the motion
        including a floor

        vertices (num_frames, num_verts, 3)
        faces (num_faces, 3)
        translations (num_frames, 3)
        cam_pose (4, 4)
        si_prec: set negative to disable si calculation
    """
    show_si = calc_si_prec > 0.0

    num_frames = root_orient.shape[0]
    ### Obtain meshes ###
    smpl_layer = smplx.SMPLHLayer("./body_models/smplh_merged/SMPLH_NEUTRAL.pkl").to(device)
    faces = torch.tensor(smpl_layer.faces.astype(np.int64),
                        dtype=torch.long,
                        device=device)
    
    smpl_meshes = smpl_layer(betas=smpl_shape,
                            global_orient=root_orient,
                            body_pose=body_pose)
    
    # Get extends of the motion
    verts_trans = smpl_meshes.vertices + translations.unsqueeze(1)
    verts_trans = (Rot @ verts_trans.flatten(0,1).detach().cpu().numpy().T).T

    # Setup scene
    scene_all, scene_sis, cam = setup_scene(verts_trans)

    ### Return frames ### -> Video generation out of method in case multiple motions are concatenated
    flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA

    trans = translations.detach().cpu().numpy()

    # render meshes
    frames = []
    for i in tqdm(range(num_frames)):
        vrts = smpl_meshes.vertices[i]

        # Determine inside triangles
        if show_si:
            surface_triangles = mesh2sdf.trimmesh_gpu(vrts[faces])
        else:
            surface_triangles = torch.ones(faces.shape[0], dtype=torch.bool, device=device)

        surface_ids = torch.where(surface_triangles)[0].cpu().numpy()
        mesh_out = trimesh.Trimesh(vrts.squeeze(0).cpu().numpy(),faces[surface_ids].cpu().numpy())

        trafo = np.eye(4)
        trafo[:3,:3] = Rot
        trafo[:3, 3] = Rot @ trans[i]

        # Render
        pr_skin = pyrender.Mesh.from_trimesh(mesh_out, material=mat_skin)

        out = ()
        for cam_pose, width, height in cam_settings:
            human_out = scene_all.add(pr_skin, name='human_skin', pose=trafo)
            cam_sce = scene_all.add(cam, name='cam', pose=cam_pose, parent_node=human_out)

            r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            c,_ = r.render(scene_all, flags=flags)
            r.delete()

            scene_all.remove_node(cam_sce)
            scene_all.remove_node(human_out)

            if show_si:
                inside_ids = torch.where(~surface_triangles)[0].cpu().numpy()
                mesh_in = trimesh.Trimesh(vrts.squeeze(0).cpu().numpy(), faces[inside_ids].cpu().numpy())

                pr_si = pyrender.Mesh.from_trimesh(mesh_in, material=mat_si)

                pr_skin.is_visible = False
                human_out = scene_sis.add(pr_skin, name='human_skin', pose=trafo)
                human_in = scene_sis.add(pr_si, name='human_si', pose=trafo)
                cam_sce_si = scene_sis.add(cam, name='cam', pose=cam_pose, parent_node=human_out)

                r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
                c_in,_ = r.render(scene_sis, flags=flags)
                r.delete()

                scene_sis.remove_node(cam_sce_si)
                scene_sis.remove_node(human_out)
                scene_sis.remove_node(human_in)

                # Compose images
                c_out = Image.fromarray(c).convert('RGBA')
                _,_,_,alpha_out = c_out.split()
                alpha_out = alpha_out.point(lambda p:p*0.75)
                c_out.putalpha(alpha_out)
                c_in = Image.fromarray(c_in).convert('RGBA')
                c = Image.alpha_composite(c_in, c_out)

            out += (cv2.cvtColor(np.array(c), cv2.COLOR_RGB2BGR),)

        frames.append(out)

    if show_si:
        meshes = smpl_layer(betas=smpl_shape)
        mc, ms = get_mesh_normalization(meshes, smpl_layer.faces.astype(np.int32))        
        verts = smpl_meshes.vertices - mc
        verts *= ms

        si_metric_mb, _ = calculate_selfintersection_volume(verts, smpl_layer.faces.astype(np.int32), prec=calc_si_prec)
        sis = np.mean(si_metric_mb)
    else:
        sis = -1

    return frames, sis

def get_prompt_slide(prompt):
    prompt_slide = Image.fromarray(np.array(Image.new('RGB', (1920, 1080), color = (255,255,255))))
    draw = ImageDraw.Draw(prompt_slide)
    prompt_split = prompt.split(" ")
    idx_prompt = 0

    while len(prompt_split) > 0:
        words = len(prompt_split)
        if words > 7:
            words = 7

        curr_prompt = prompt_split[:words]
        prompt_split = prompt_split[words:]
        draw.text((230,200+idx_prompt*80), f"{' '.join(curr_prompt)}", fill=(0,0,0,255), font=font2)
        idx_prompt += 1

    return np.array(prompt_slide)

def add_prompt_to_render(img, prompt, si_val, method_name):
    p_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
    draw = ImageDraw.Draw(p_img)
    draw.text((25,1000), f"SI={int(si_val)} cm3", fill=(255,0,0,255), font=font)
    draw.text((1520, 950), f"Method: {method_name.split(' ')[0]}", fill=(0,0,0,255), font=font)
    if method_name.split(' ')[1] == "(Ours)":
        draw.text((1520, 1000), f"{method_name.split(' ')[1]}", fill=(0,0,0,255), font=font, stroke_width=2, stroke_fill="black")
    else:
        draw.text((1520, 1000), f"{method_name.split(' ')[1]}", fill=(0,0,0,255), font=font)
    
    prompt_split = prompt.split(" ")
    idx_prompt = 0
    while len(prompt_split) > 0:
        words = len(prompt_split)
        if words > 14:
            words = 14

        curr_prompt = prompt_split[:words]
        prompt_split = prompt_split[words:]
        draw.text((25,25+idx_prompt*50), f"{' '.join(curr_prompt)}", fill=(0,0,0,255), font=font)
        idx_prompt += 1

    return cv2.cvtColor(np.array(p_img), cv2.COLOR_RGB2BGR)


def add_text_to_method_compare_render(img, prompt, si_baseline, si_sia, method_name_baseline, method_name_sia):
    p_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
    draw = ImageDraw.Draw(p_img)
    draw.text((25, 1015), f"Method: {method_name_baseline}", fill=(0,0,0,255), font=font)
    if si_baseline >= 0.0:
        draw.text((25, 965), f"SI={int(si_baseline)} cm3", fill=(255,0,0,255), font=font)
    draw.text((1300, 1015), f"Method: {method_name_sia}", fill=(0,0,0,255), font=font)          #MDM: 1375, MoMask: 1300
    if si_sia >= 0.0:
        draw.text((1625, 965), f"SI={int(si_sia)} cm3", fill=(255,0,0,255), font=font)
    
    prompt_split = prompt.split(" ")
    idx_prompt = 0
    while len(prompt_split) > 0:
        words = len(prompt_split)
        if words > 14:
            words = 14

        curr_prompt = prompt_split[:words]
        prompt_split = prompt_split[words:]
        draw.text((25,25+idx_prompt*50), f"{' '.join(curr_prompt)}", fill=(0,0,0,255), font=font)
        idx_prompt += 1

    return cv2.cvtColor(np.array(p_img), cv2.COLOR_RGB2BGR)

mat_skin = pyrender.MetallicRoughnessMaterial("skin", baseColorFactor=[0.8, 0.352, 0.160, 1.0], alphaMode='OPAQUE')
mat_si = pyrender.MetallicRoughnessMaterial("SI", baseColorFactor=[1.0, 0.0, 0.0, 1.0], alphaMode='OPAQUE')
Rot = Ry(90)@Rz(90)

if __name__ == "__main__":
    args = get_args()
    h2s_layer = HumanML3D_To_SMPL_Layer(smpl_rotation_representation="rot_mat",
                                        dataset=args.dataset).to(device)

    baseline_string = args.baseline_name + " (Baseline)"
    sia_string = args.sia_name + " (Ours)"

    results_base = np.load(args.motion_path_baseline, allow_pickle=True)[()]
    results_sia = np.load(args.motion_path_sia, allow_pickle=True)[()]

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tmp"), exist_ok=True)
    frmcnt = 0


    ### Pyrender Setup
    cam_pose1 = np.eye(4)
    cam_pose1[:3,:3] = Rx(-20)
    cam_pose1[:3,3] = [0.0, 0.75, 2.5]
    cam_pose2 = np.eye(4)
    cam_pose2[:3,:3] = Rx(-90)
    cam_pose2[:3,3] = [0.0, 2.0, 0.0]
    cam_pose3 = np.eye(4)
    cam_pose3[:3,:3] = Rz(20) @ Ry(90)
    cam_pose3[:3,3] = [2.0, 0.5, 0.0]

    cams = [(cam_pose1, 1920, 1080)]#, (cam_pose2, 960, 540), (cam_pose3, 960, 540)]

    title = np.array(Image.open("visualize/VideoTitleSlide.png"))
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",size=40)
    font2 = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",size=60)
    white = np.array(Image.new('RGB', (1920, 1080), color = (255,255,255)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_frame_buffer = []

    #out_frame_buffer += 300*[title]
    """"
    # Arm cross motion
    cross_mdm = np.load("/home/hpr2hi/mdm_fork/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000475000_seed10_Tightly_cross_arms_and_open_them_again/results.npy", allow_pickle=True)[()] # idx 0
    cross_siamdm = np.load("/home/hpr2hi/mdm_fork/save/CLUSTER_SAVE/CA-MDM-sphere-l2-si001/samples_CA-MDM-sphere-l2-si001_000600000_seed10_Tightly_cross_arms_and_open_them_again/results.npy", allow_pickle=True)[()] # idx 1

    motion_mdm = torch.from_numpy(np.array(cross_mdm['hml'][0:1])).to(device).permute([0,2,3,1]).float()
    motion_siamdm = torch.from_numpy(np.array(cross_siamdm['hml'][1:2])).to(device).permute([0,2,3,1]).float()
    prompt = cross_mdm['text'][0]

    # Add prompt
    prompt_slide = get_prompt_slide(prompt)
    #out_frame_buffer += 60*[prompt_slide]

    smpl_shape = torch.tensor([params.beta_hml]).expand([motion_mdm.shape[3],-1]).to(device)

    front_baseline = []
    front_sia = []
    si_baseline = -1
    si_sia = -1
    for motion, method in [(motion_mdm, baseline_string), (motion_siamdm, sia_string)]:
        smpl_params = h2s_layer(motion)

        frames, sis = render_smpl_motion(smpl_shape,
                           smpl_params['root_orient'].flatten(0,1),
                           smpl_params['body_orient'].flatten(0,1),
                           smpl_params['trans'].flatten(0,1),
                           cams,
                           args.si_prec)

        if "Baseline" in method:
            for frame in frames:
                front_baseline.append(frame[0])
            si_baseline = sis
        else:
            for frame in frames:
                front_sia.append(frame[0])
            si_sia = sis
        ### Render front, top and side view
        #for front, top, side in frames:
        #    f_right = np.concatenate((top, side), axis=0)
        #    f_coll = np.concatenate((front[:,480:1440], f_right), axis=1)

        #    f_txt = add_prompt_to_render(f_coll, prompt, -1, method)

        #    out_frame_buffer.append(f_txt)


    for frm in range(len(front_baseline)):
        f_coll = np.concatenate((front_baseline[frm][:,480:1440], front_sia[frm][:,480:1440]), axis=1)

        f_txt = add_text_to_method_compare_render(f_coll, prompt, si_baseline, si_sia, baseline_string, sia_string)

        out_frame_buffer.append(f_txt)
        cv2.imwrite(os.path.join(args.out_dir, "tmp",f"{frmcnt}.png"),f_txt)
        frmcnt += 1
    
    for _ in range(5):
        out_frame_buffer += [white]
        cv2.imwrite(os.path.join(args.out_dir, "tmp",f"{frmcnt}.png"),white)
        frmcnt += 1        

    out = cv2.VideoWriter(os.path.join(args.out_dir, f"Qualitative_Results_n1.avi"), fourcc=fourcc, fps=20, frameSize=(1920,1080))

    for frm in tqdm(out_frame_buffer, desc="Writing video"):
        out.write(frm)
    """
    # Test set
    # MDM exclude ids
    #exclude_ids = [0, 1, 3, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 22, 23, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62]
    # MoMask exclude ids
    #exclude_ids = [1, 2, 3, 4, 5, 7, 9, 10, 11, 14, 16, 17, 21, 22, 23, 24, 28, 29, 30, 31, 33, 34, 36, 44, 46, 48, 50, 51, 52, 54, 56, 57, 61, 62, 63, 64]
    # MoMask kit exclude ids
    exclude_ids = [0, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 32, 35, 36, 39, 40, 42, 43, 44, 45, 47, 50, 54, 55,56,57, 58, 59,61, 63]
    for motion_id in tqdm(range(len(results_base['hml']))):
        if motion_id in exclude_ids:
            continue
       
        motion_mdm = torch.from_numpy(np.array(results_base['hml'][motion_id:motion_id+1])).float().to(device).permute([0,2,3,1])
        motion_siamdm = torch.from_numpy(np.array(results_sia['hml'][motion_id:motion_id+1])).float().to(device).permute([0,2,3,1])
        prompt = results_base['text'][motion_id]

        motion_mdm = motion_mdm[..., :results_base['lengths'][motion_id]]
        motion_siamdm = motion_siamdm[..., :results_sia['lengths'][motion_id]]

        # Add prompt
        prompt_slide = get_prompt_slide(prompt)

        for _ in range(60):
            out_frame_buffer += [prompt_slide]
            cv2.imwrite(os.path.join(args.out_dir, "tmp",f"{frmcnt}.png"),prompt_slide)
            frmcnt += 1    

        if args.dataset == 'hml':
            smpl_shape = torch.tensor([params.beta_hml]).expand([motion_mdm.shape[3],-1]).to(device)
        elif args.dataset == 'kit':
            smpl_shape = torch.tensor([params.beta_kit]).expand([motion_mdm.shape[3],-1]).to(device)
        else:
            raise NotImplementedError

        front_baseline = []
        front_sia = []
        si_baseline = -1
        si_sia = -1
        for motion, method in [(motion_mdm, baseline_string), (motion_siamdm, sia_string)]:
            smpl_params = h2s_layer(motion)

            frames, sis = render_smpl_motion(smpl_shape,
                            smpl_params['root_orient'].flatten(0,1),
                            smpl_params['body_orient'].flatten(0,1),
                            smpl_params['trans'].flatten(0,1),
                            cams,
                            args.si_prec)

            if "Baseline" in method:
                for frame in frames:
                    front_baseline.append(frame[0])
                si_baseline = sis
            else:
                for frame in frames:
                    front_sia.append(frame[0])
                si_sia = sis

        for frm in range(len(front_baseline)):
            f_coll = np.concatenate((front_baseline[frm][:,480:1440], front_sia[frm][:,480:1440]), axis=1)
            f_txt = add_text_to_method_compare_render(f_coll, prompt, si_baseline, si_sia, baseline_string, sia_string)

            out_frame_buffer.append(f_txt)
            cv2.imwrite(os.path.join(args.out_dir, "tmp",f"{frmcnt}.png"),f_txt)
            frmcnt += 1   

        #for _ in range(5):
        #    out_frame_buffer += [white]
        #    cv2.imwrite(os.path.join(args.out_dir, "tmp",f"{frmcnt}.png"),white)
        #    frmcnt += 1

        if motion_id%5 == 0:
           out = cv2.VideoWriter(os.path.join(args.out_dir, f"Qualitative_Results_{motion_id}.avi"), fourcc=fourcc, fps=20, frameSize=(1920,1080))

           for frm in tqdm(out_frame_buffer, desc="Writing video"):
               out.write(frm)


    out = cv2.VideoWriter(os.path.join(args.out_dir, "Qualitative_Results.avi"), fourcc=fourcc, fps=20, frameSize=(1920,1080))

    for frm in tqdm(out_frame_buffer, desc="Writing video"):
        out.write(frm)
        
# python -m visualize.create_video --motion_path_baseline ~/momask_fork/generation/eval_base/results.npy --baseline_name MoMask --motion_path_sia ~/momask_fork/generation/eval_sia/results.npy --sia_name SIA-MoMask --out_dir ~/momask_fork/generation/ --si_prec 0.06