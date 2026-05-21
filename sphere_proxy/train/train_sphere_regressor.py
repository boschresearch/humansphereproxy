## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
import shutil
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sphere_proxy.data_loader.sdf_dataset import SDFDataset
from sphere_proxy.models.sphere_regressor import SphereRegressor
from sphere_proxy.models.losses import clamped_outside_SDF_loss, emptiness_loss, selfintersection_loss, joint_distribution_loss, radius_loss
from sphere_proxy.util.util import get_config, fixseed

device = "cuda" if torch.cuda.is_available() else "cpu"

def calc_sdf(points, centers, radii):
    """Evaluate the sdf of the sphere proxy at a given point.


        // The following fct is from DualSDF
        //   (https://github.com/zekunhao1995/DualSDF/blob/master/models/sdfsphere.py)
        // Copyright (c) 2020 Zekun Hao, licensed under the MIT license,
        // cf. 3rd-party-licenses.txt file in the root directory of this source tree.

        Params:
            points: points where to evaluate sdf (bs, num_points, 3)
    """
    c = centers.unsqueeze(-3) # (bs, 1, num_spheres, 3)
    r = radii.unsqueeze(-2) # (bs, 1, num_spheres)
    p = points.unsqueeze(-2) # (bs, num_points, 1, 3)

    dists = torch.norm(p - c, dim=-1) - r # (bs, num_points, num_spheres)
    # TODO: Here we can obtain the sphere corresponding to the sdf value.
    # This can be used to apply the weighting for the arms and legs.
    sdf, _ = torch.min(dists, dim=-1)  # (bs, num_points)

    return sdf, dists

def train_loop(data_loader, model, optim, config):
    train_loss = 0
    model.train()

    for batch, data in enumerate(tqdm.tqdm(data_loader)):
        smpl_shape = data['shape'].to(device)
        gt_sdfs = data['sdfs'].to(device)
        joints = data['joints'].to(device)
        joint_labels = data['joint_labels']

        sdf_weights = torch.Tensor(config['joint_importance_weights'])[joint_labels].to(device)

        sphere_params = model(smpl_shape)
        pred_sdfs, dists = calc_sdf(gt_sdfs[:,:,:3], sphere_params[:,:,1:], torch.exp(sphere_params[:,:,0]))

        loss = clamped_outside_SDF_loss(pred_sdfs, gt_sdfs[:,:,3], sdf_weights)

        if config['training']['sphere']['lambda_emptiness'] != 0:
            loss += float(config['training']['sphere']['lambda_emptiness']) * emptiness_loss(dists, gt_sdfs[:,:,3])

        if config['training']['sphere']['lambda_intersect'] != 0:
            loss += float(config['training']['sphere']['lambda_intersect']) * selfintersection_loss(sphere_params[:,:,1:], torch.exp(sphere_params[:,:,0]))

        if config['training']['sphere']['lambda_dist'] != 0:
            # Get weights and normalize them
            weights = torch.Tensor(config['joint_importance_weights']).to(device)
            #weights /= torch.sum(weights)
            loss += float(config['training']['sphere']['lambda_dist']) * joint_distribution_loss(sphere_params[:,:,1:], joints, weights)

        if config['training']['sphere']['lambda_rad'] != 0:
            loss += float(config['training']['sphere']['lambda_rad']) * radius_loss(torch.exp(sphere_params[:,:,0]), config['min_rad'])

            
        with torch.no_grad():
            train_loss += loss.item()

        loss.backward()
        optim.step()
        optim.zero_grad()


    train_loss /= len(data_loader)
    return train_loss


def val_loop(data_loader, model, config):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data in tqdm.tqdm(data_loader):
            smpl_shape = data['shape'].to(device)
            gt_sdfs = data['sdfs'].to(device)
            joints = data['joints'].to(device)

            sphere_params = model(smpl_shape)
            pred_sdfs, dists = calc_sdf(gt_sdfs[:,:,:3], sphere_params[:,:,1:], torch.exp(sphere_params[:,:,0]))

            loss = clamped_outside_SDF_loss(pred_sdfs, gt_sdfs[:,:,3], torch.ones_like(pred_sdfs))

            if config['training']['sphere']['lambda_emptiness'] != 0:
                loss += float(config['training']['sphere']['lambda_emptiness']) * emptiness_loss(dists, gt_sdfs[:,:,3])

            if config['training']['sphere']['lambda_intersect'] != 0:
                loss += float(config['training']['sphere']['lambda_intersect']) * selfintersection_loss(sphere_params[:,:,1:], torch.exp(sphere_params[:,:,0]))

            if config['training']['sphere']['lambda_dist'] != 0:
                # Get weights and normalize them
                weights = torch.Tensor(config['joint_importance_weights']).to(device)
                #weights /= torch.sum(weights)
                loss += float(config['training']['sphere']['lambda_dist']) * joint_distribution_loss(sphere_params[:,:,1:], joints, weights)

            if config['training']['sphere']['lambda_rad'] != 0:
                loss += float(config['training']['sphere']['lambda_rad']) * radius_loss(torch.exp(sphere_params[:,:,0]), config['min_rad'])
            
            val_loss += loss.item()

    val_loss /= len(data_loader)
    return val_loss
     

if __name__ == "__main__":
    print(f"Run training on {device}...")
    config, config_path = get_config()
    if config['seed'] is not None:
        print(f"Set seed {config['seed']}")
        fixseed(int(config['seed']))

    # Print loss weights
    if config['training']['sphere']['lambda_emptiness'] != 0:
        print(f"Use emptiness loss - weight {config['training']['sphere']['lambda_emptiness']}")

    if config['training']['sphere']['lambda_intersect'] != 0:
        print(f"Use intersection loss - weight {config['training']['sphere']['lambda_intersect']}")

    if config['training']['sphere']['lambda_dist'] != 0:
        print(f"Use distribution loss - weight {config['training']['sphere']['lambda_dist']}")

    if config['training']['sphere']['lambda_rad'] != 0:
        print(f"Use radii loss - weight {config['training']['sphere']['lambda_rad']}")

    # Setup logdir
    log_dir = os.path.join(config['logging']['log_dir'], "SR")
    os.makedirs(log_dir)
    shutil.copy(config_path, config['logging']['log_dir'])

    if config['logging']['tb_logging']:
        tb_dir = os.path.join(log_dir, "tb")
        os.makedirs(tb_dir)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"Save tensorboard logging to {tb_dir} ...")

    ### Setup data
    print("Loading data...")
    train_path = os.path.join(config['data']['path'], "train.txt")
    val_path = os.path.join(config['data']['path'], "val.txt")

    train_data = SDFDataset(train_path,
                            num_sdf_samples=config['data']['num_samples'],
                            perc_sphere=config['data']['perc_sphere'],
                            perc_detail=config['data']['perc_detail'])
    train_dataloader = DataLoader(train_data,
                                  batch_size=config['training']['sphere']['batch_size'],
                                  num_workers=config['training']['sphere']['num_workers'],
                                  shuffle=True)
    
    val_data = SDFDataset(val_path,
                          num_sdf_samples=config['data']['num_samples'],
                          perc_sphere=config['data']['perc_sphere'],
                          perc_detail=config['data']['perc_detail'])
    val_dataloader = DataLoader(val_data,
                                  batch_size=config['training']['sphere']['batch_size'],
                                  num_workers=config['training']['sphere']['num_workers'],
                                  shuffle=True)

    ### Setup model
    print("Creating model...")
    sphereRegressor = SphereRegressor(config['model']['num_spheres'],
                          config['model']['smpl_shape_dim'],
                          config['model']['latent_dim'])
    sphereRegressor = sphereRegressor.to(device)
    
    ### Setup optimization
    optim = torch.optim.Adam(sphereRegressor.parameters(), lr=float(config['training']['sphere']['lr']))
    lr_schedule = torch.optim.lr_scheduler.StepLR(optim,
                                                  step_size=config['training']['sphere']['lr_schedule']['step'],
                                                  gamma=config['training']['sphere']['lr_schedule']['factor'])

    ### Optimization loop
    print("Starting training...")
    for epoch in range(config['training']['sphere']['epochs']):
        print(f"Epoch {epoch+1}")
        t_loss = train_loop(train_dataloader, sphereRegressor, optim, config)
        v_loss = val_loop(val_dataloader, sphereRegressor, config)

        print(f"Avg train loss: {t_loss}")
        print(f"Avg val loss: {v_loss}")


        if writer:
            writer.add_scalar("Loss train", t_loss, epoch)
            writer.add_scalar("Loss val", v_loss, epoch)

        lr_schedule.step()

        
    print("Done!")
    print(f"Saving model to {log_dir}...")
    torch.save(sphereRegressor.state_dict(), os.path.join(log_dir, "sphere_regressor.pth"))