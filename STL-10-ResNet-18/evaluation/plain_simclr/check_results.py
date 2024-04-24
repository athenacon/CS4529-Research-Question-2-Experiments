import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import neptune

from simclr.modules import get_resnet
from torch.utils.data import Dataset, random_split
from utils import yaml_config_hook

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
from torch.utils.data import Subset

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    ckpt_path = os.path.join("checkpoints_linear_evaluation", filename)
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
    
     
    normalize = transforms.Normalize(mean=[0.4914, 0.4823,0.4466],
                                        std=[0.247, 0.243, 0.261]) 

    test_transform = transforms.Compose(
    [
        transforms.CenterCrop((96 * 0.875, 96 * 0.875)),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        normalize
    ]
    )
    
    test_dataset = torchvision.datasets.STL10(
        args.dataset_dir,
        split="test",
        download=True,
        transform=test_transform,
    )

    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
        
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)

    print(f"Test size: {len(test_dataset)}")
     
    num_test = len(test_loader.dataset)
    print(f"Number of test samples: {num_test}")

    from simclr.modules.resnet_hacks import modify_resnet_model

    encoder_previous = get_resnet(args.resnet, pretrained=False)
    encoder = modify_resnet_model(encoder_previous, cifar_stem=True, v1=True)
    encoder.fc = torch.nn.Identity()

    import torch.nn as nn
    head = nn.Linear(512, 10) 
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(encoder, head)
    model.cuda(args.device)
    checkpoint = torch.load("check_results_checkpoints/simclr_linear_evaluation_after_pretrained_ckpt_epoch_100.pth.tar")    
    model.load_state_dict(checkpoint['model_state'])
    correct = 0
    model.eval()
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
    
    # Testing fine-tune
    correct = 0
    checkpoint = torch.load("check_results_checkpoints/simclr_finetune_ckpt_epoch_20.pth.tar")    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
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
    
 