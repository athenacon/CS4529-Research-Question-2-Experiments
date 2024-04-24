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
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import neptune
from simclr import SimCLR
from simclr.modules import get_resnet

from utils import yaml_config_hook
from torch.utils.data import random_split, Dataset
from torchvision import datasets, transforms
from torch import nn, optim

from simclr.modules.resnet_hacks import modify_resnet_model
# Capsule Network
from capsule_network import resnet20
from utilscapsnet import AverageMeter

class ApplyTransform(Dataset):
    # reference:https://stackoverflow.com/questions/56582246/correct-data-loading-splitting-and-augmentation-in-pytorch
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        if transform is None and target_transform is None:
            print("Transforms have failed")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)
    
def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "simclr_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)
        
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")      
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                     std=[0.247, 0.243, 0.261])
        
    
    test_transform = transforms.Compose(
    [
        transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
        transforms.Resize((32)),
        transforms.ToTensor(),
        normalize
    ]
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,

    )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)
    print("Length of test_loader", len(test_loader))
    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    modified_resnet = modify_resnet_model(encoder)
    capsule_network = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="SR").to(args.device)
    
    # initialize model
    model = SimCLR(modified_resnet, capsule_network)
    device = args.device
    checkpoint = torch.load("check_results_checkpoints/simclr_linear_evaluation_after_pretrained_ckpt_epoch_100.pth.tar", map_location=device)  
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model = model.to(args.device)
    # print(model) 
    print("Model loaded from pre-trained model successfully")
   
    # Testing
    correct = 0
    model.eval()
    num_test = len(test_loader.dataset)
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(args.device), y.to(args.device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (num_test)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test, perc, error)
    )
    
    # Testing fine-tuning
    correct = 0
    model.eval()
    checkpoint = torch.load("check_results_checkpoints/simclr_finetune_1_ckpt_epoch_20.pth.tar", map_location=device)  
    model.load_state_dict(checkpoint['model_state'], strict=True)
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(args.device), y.to(args.device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (num_test)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test, perc, error)
    )