## Copyright (c) 2025 Robert Bosch GmbH
## SPDX-License-Identifier: AGPL-3.0

import os
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Own classes
from sphere_proxy.data_loader.shape_joint_dataset import ShapeJointDataset
from sphere_proxy.models.joint_regressor import JointRegressor
from sphere_proxy.util.util import get_config, fixseed


device = "cuda" if torch.cuda.is_available() else "cpu"

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss = 0
    model.train()

    for batch, (data, label) in enumerate(dataloader):
        data = data.to(device)
        label = label.to(device)
        # Compute loss
        pred = model(data)
        pred = pred.reshape(data.shape[0], dataloader.dataset.num_joints, 3)
        loss = loss_fn(pred, label)

        with torch.no_grad():
            train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if batch % 25 == 0:
            loss, current = loss.item(), batch * len(data) + len(data)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        
    train_loss /= len(dataloader)
    print(f"Train Error: \nAvg loss: {train_loss:>8f}\n")

    return train_loss


def val_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            pred = pred.reshape(data.shape[0], dataloader.dataset.num_joints, 3)
            test_loss += loss_fn(pred, label).item()

    test_loss /= num_batches
    print(f"Val Error: \nAvg loss: {test_loss:>8f}\n")

    return test_loss


def eval_and_log(train_loader, val_loader, test_loader, log_dir, model, loss_fn):
    model.eval()
    train_loss = 0
    val_loss = 0
    test_loss = 0

    with torch.no_grad():
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            pred = pred.reshape(data.shape[0], train_loader.dataset.num_joints, 3)
            train_loss += loss_fn(pred, label).item()

        train_loss /= len(train_loader)

        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            pred = pred.reshape(data.shape[0], val_loader.dataset.num_joints, 3)
            val_loss += loss_fn(pred, label).item()

        val_loss /= len(val_loader)

        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            pred = pred.reshape(data.shape[0], test_loader.dataset.num_joints, 3)
            test_loss += loss_fn(pred, label).item()

        test_loss /= len(test_loader)

    eval_str = ""
    eval_str+= f"------ EVAL------\n"
    eval_str+=f"Avg train loss: {train_loss}\n"
    eval_str+=f"Avg val loss: {val_loss}\n"
    eval_str+=f"Avg test loss: {test_loss}\n"
    print(eval_str)

    with open(os.path.join(log_dir, "eval.txt"), "w") as evalfile:
        evalfile.write(eval_str)




if __name__ == "__main__":
    config, config_path = get_config()
    if config['seed'] is not None:
        print(f"Set seed {config['seed']}")
        fixseed(int(config['seed']))
    
    print(f"Run training on {device}")
    log_dir = os.path.join(config['logging']['log_dir'], "JR")
    os.makedirs(log_dir)
    shutil.copy(config_path, config['logging']['log_dir'])

    ### Setup logger ###
    if config['logging']['tb_logging']:
        tb_dir = os.path.join(log_dir, "tb")
        os.makedirs(tb_dir)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"Save tensorboard logging to {tb_dir} ...")



    ### Setup data ###
    print("Loading data ...")
    train_path = os.path.join(config['data']['path'], "train.txt")
    val_path = os.path.join(config['data']['path'], "val.txt")
    test_path = os.path.join(config['data']['path'], "test.txt")

    # Train data
    training_data = ShapeJointDataset(train_path)
    train_dataloader = DataLoader(training_data,
                                  batch_size=config['training']['joint']['batch_size'],
                                  shuffle=True,
                                  num_workers=config['training']['joint']['num_workers'])
    # Val data
    val_data = ShapeJointDataset(val_path)
    val_dataloader = DataLoader(val_data,
                                batch_size=config['training']['joint']['batch_size'],
                                shuffle=False,
                                num_workers=config['training']['joint']['num_workers'])
    # Test data
    test_data = ShapeJointDataset(test_path)
    test_dataloader = DataLoader(test_data,
                                 batch_size=config['training']['joint']['batch_size'],
                                 shuffle=False,
                                 num_workers=config['training']['joint']['num_workers'])

    ### Setup model ###
    print("Creating model ...")
    model = JointRegressor(config['model']['smpl_shape_dim'],
                           config['model']['num_joints'],
                           config['model']['latent_dim']).to(device)

    ### Setup optimization
    loss_fkt = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),
                             lr=float(config['training']['joint']['lr']))
    if config['training']['joint']['lr_schedule'] is not None:
        print("Use LR schedule ...")
        schedule = torch.optim.lr_scheduler.StepLR(optim,
                                                  step_size=config['training']['joint']['lr_schedule']['step'],
                                                  gamma=config['training']['joint']['lr_schedule']['factor'])

    ### Optimization loop
    print("Starting training ...")
    for epoch in range(config['training']['joint']['epochs']):
        print(f"Epoch {epoch+1}\n -------------")
        t_loss = train_loop(train_dataloader, model, loss_fkt, optim)
        v_loss = val_loop(val_dataloader, model, loss_fkt)

        if schedule:
            schedule.step()

        if writer:
            writer.add_scalar("Loss train", t_loss, epoch)
            writer.add_scalar("Loss val", v_loss, epoch)

        if schedule is not None:
            if ((epoch+1) % config['training']['joint']['lr_schedule']['step']) == 0:
                print("Learning rate step...")



    print("Done!")
    print(f"Saving model to {log_dir} ...")
    torch.save(model.state_dict(), os.path.join(log_dir, "joint_regressor.pth"))
    eval_and_log(train_dataloader, val_dataloader, test_dataloader, log_dir, model, loss_fkt)
