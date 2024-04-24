# ref https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/7
seed_value = 42
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `pytorch` pseudo-random generator at a fixed value
import torch
torch.manual_seed(seed_value)

import os
import numpy as np
import torch
import torchvision
import argparse
import neptune

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, resnet
from simclr.modules.transformations import TransformsSimCLR
from model import load_optimizer, save_model
from utils import yaml_config_hook

def train(args, train_loader, model, criterion, optimizer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
 
        loss_epoch += loss.item()
    return loss_epoch


def main(args):

    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(),
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers 
    )

    # initialize ResNet20
    encoder = resnet.resnet20()
    n_features = encoder.linear.in_features  # get dimensions of fc layer
    print("Feature size of encoder:", n_features)
    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, 1)

    model = model.to(args.device)
    
    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
         
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer)

        scheduler.step()

        # save lr
        # run["pre-training/epoch/lr"].log(lr)
        # run["pre-training/epoch/loss"].log(loss_epoch/len(train_loader))
 
    ## end training
    save_model(args, model, optimizer)
    # run.stop()


if __name__ == "__main__":
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.world_size = 1 # single GPU
    main(args)
