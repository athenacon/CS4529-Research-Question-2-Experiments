from pathlib import Path
import argparse
import os
import random
import neptune
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import random_split, Dataset
from utilscapsnet import AverageMeter
import torch 
import resnet
from capsule_network import resnet20 
import numpy as np

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
        default=0,
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

    # load the pre-trained model
    model = VICReg(args).cuda(gpu)
    print(model) 
    checkpoint = torch.load("check_results_checkpoints/vicreg_linear_evaluation_after_pretrained_ckpt_epoch_100.pth.tar", map_location='cuda')  # or 'cpu'
    model.load_state_dict(checkpoint['model_state'], strict=True)
    # print(model)
    print("Model loaded") 
    model.to(gpu) 
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
        "datasets",
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
    
    # Testing fine-tuned model
    checkpoint = torch.load("check_results_checkpoints/vicreg_linear_evaluation_after_pretrained_ckpt_epoch_20.pth.tar", map_location='cuda')  # or 'cpu'
    model.load_state_dict(checkpoint['model_state'], strict=True)
    
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
    
def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "vicreg_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)
  
class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = 2048
        self.backbone, _ = resnet.__dict__[args.arch](
            zero_init_residual=True 
        )
        self.projection_head = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="SR").to("cuda")

    def forward(self, x ):
        x = self.projection_head(self.backbone(x))
        return x

if __name__ == "__main__":
    main()
