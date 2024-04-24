from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import neptune
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset
from utilscapsnet import AverageMeter
import torch
from norb import smallNORB
from torch.utils.data import Subset
import resnet
from capsule_network import resnet20 
import numpy as np
from torch.utils.data import Subset

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
    # parser.add_argument(
    #     "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    # )

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

def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):

    data_dir = data_dir + '/' + dataset 
    from torchvision.transforms import InterpolationMode
    trans_train = [  
                transforms.Resize(48, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(32),
                transforms.ColorJitter(brightness=32./255, contrast=0.3),
                transforms.ToTensor(),
            ]
    
    trans_valid = [
            transforms.Resize(48, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
    ] 
    dataset = smallNORB(data_dir, train=True, download=True,
                    transform = None)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[split:]
    valid_idx = indices[:split]

    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)

    train_set_apply_transf = ApplyTransform(train_set, transform=transforms.Compose(trans_train))
    val_set_apply_transf = ApplyTransform(valid_set, transform=transforms.Compose(trans_valid))

    train_loader = torch.utils.data.DataLoader(
        train_set_apply_transf, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_set_apply_transf, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    print("type of train_loader", type(train_loader))
    print("type of valid_loader", type(valid_loader))
    print("Length of train_loader", len(train_loader))
    print("Length of valid_loader", len(valid_loader))
    
    return train_loader, valid_loader


def get_test_loader(data_dir,
                    dataset,
                    batch_size, 
                    num_workers=4,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    from torchvision.transforms import InterpolationMode
    if dataset == "smallNorb":
        trans = [
                   #  During test, we crop a 32 Ã— 32 patch from the center of the image. Matrix capsules
            # with em routing
                transforms.Resize(48, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                 ]
        dataset = smallNORB(data_dir, train=False, download=True,
                                transform=transforms.Compose(trans))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    print("type of test_loader", type(data_loader))
    print("Length of test_loader", len(data_loader))
    return data_loader


def main_worker(gpu, args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
 

    torch.backends.cudnn.benchmark = True

    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")
    
    # load the pre-trained model
    model = VICReg(args).cuda(gpu)
    checkpoint = torch.load("pre_trained_model_epoch_1000.pth", map_location='cuda')  # or 'cpu'
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("Model loaded")  

    for param in model.parameters():
        param.requires_grad = False
 
    for param in model.projection_head.fc.parameters():
        param.requires_grad = True
    
    model.to(gpu) 
   
    kwargs = {'num_workers': 0, 'pin_memory': False}

    data_loader = get_train_valid_loader(
        "./data", "smallNorb", args.batch_size,
        42, "full", 0.1,
        "True", **kwargs
    )
    train_loader = data_loader[0]
    valid_loader = data_loader[1]
    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)  
    
    print("Data loaders created successfully")
    print("Length of training dataset", len(train_loader.dataset))
    print("Length of validation dataset", len(valid_loader.dataset))
    
    # now we import testing dataset
    test_loader = get_test_loader("./data", "smallNorb", args.batch_size, **kwargs)
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", num_test)
    
    start_epoch = 0
    criterion = nn.NLLLoss().to(gpu)
    optimizer = optim.SGD(model.projection_head.fc.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print("\n[*] Train on {} samples, validate on {} samples".format(
        num_train, num_valid)
    )
    
    best_valid_acc = 0
    for epoch in range(start_epoch, args.epochs):
        # get current lr
        for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
            break
        model.eval()
        model.projection_head.fc.train()
        model.backbone.eval()

        losses = AverageMeter()
        accs = AverageMeter()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(gpu), y.to(gpu)
 
            out = model(x)
            
            loss = criterion(out, y)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct = (pred == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

            # compute gradients and update SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_acc = losses.avg, accs.avg
        # run["after_pretraining/training/epoch/loss"].log(train_loss)
        # run["after_pretraining/training/epoch/acc"].log(train_acc)
        # evaluate on validation set
        with torch.no_grad():
            
            model.eval()

            losses = AverageMeter()
            accs = AverageMeter()

            for i, (x, y) in enumerate(valid_loader):
                x, y = x.to(gpu), y.to(gpu)

                out = model(x)
                
                loss = criterion(out, y)

                # compute accuracy
                pred = torch.max(out, 1)[1]
                correct = (pred == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])
        
        valid_loss, valid_acc = losses.avg, accs.avg

        # run["after_pretraining/validation/epoch/loss"].log(valid_loss)
        # run["after_pretraining/validation/epoch/acc"].log(valid_acc)  
        
        # decay lr
        scheduler.step()
    save_checkpoint(
        {   'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'valid_acc': valid_acc
        } 
        )
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
    
    # run["after_pretraining/testing/epoch/loss"].log(error)
    # run["after_pretraining/testing/epoch/acc"].log(perc)
    
    # run.stop()

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
    
class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = 2048
        self.backbone, _ = resnet.__dict__[args.arch](
            zero_init_residual=True 
        )
        self.projection_head = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="EM").to("cuda")


    def forward(self, x ):
        x = self.projection_head(self.backbone(x))
        return x
 


if __name__ == "__main__":
    main()
