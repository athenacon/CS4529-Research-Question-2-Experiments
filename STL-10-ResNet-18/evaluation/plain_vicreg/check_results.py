from pathlib import Path
import argparse
import os
import random
import neptune
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset 
import torch
import numpy as np
import resnet
# from capsule_network import resnet20 
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


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on CIFAR-10"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    
    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet18")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
     
    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    # single-gpu training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main_worker(device, args)

def exclude_bias_and_norm(p):
    return p.ndim == 1

def main_worker(gpu, args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    
    torch.backends.cudnn.benchmark = True
    
    backbone, _ = resnet.__dict__[args.arch](
            zero_init_residual=True
        )

    head = nn.Linear(512, 10)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)
    state_dict = torch.load("check_results_checkpoints/vicreg_linear_evaluation_after_pretrained_ckpt_epoch_100.pth.tar", map_location="cpu")
    model_state_dict = state_dict['model_state']
    model.load_state_dict(model_state_dict, strict=True)

    model.to(gpu)
    # print(model)
    
    import torchvision
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
        ".\datasets",
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
    num_test = len(test_loader.dataset)
    # Testing
    correct = 0
    model.eval()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(gpu), y.to(gpu)

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
    # Testing fine-tuned
    correct = 0
    state_dict = torch.load("check_results_checkpoints/vicreg_linear_evaluation_after_pretrained_ckpt_epoch_20.pth.tar", map_location="cpu")
    model_state_dict = state_dict['model_state']
    model.load_state_dict(model_state_dict, strict=True)
    model.to(gpu)
    model.eval()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(gpu), y.to(gpu)

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
    

def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "vicreg_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)
    
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
     
if __name__ == "__main__":
    main()
