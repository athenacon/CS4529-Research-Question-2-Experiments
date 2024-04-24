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
        
    train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    )

    test_transform = transforms.Compose(
    [
        transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
        transforms.Resize((32)),
        transforms.ToTensor(),
        normalize
    ]
    )
    full_training_dataset = datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    # transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    # Data loading code
    training_dataset_size = 45000
    val_dataset_size = 5000
    
    train_dataset, val_dataset = random_split(full_training_dataset, [training_dataset_size, val_dataset_size])
    
    train_dataset = ApplyTransform(train_dataset, transform=train_transform)
    val_dataset = ApplyTransform(val_dataset, transform=test_transform)
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,

    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)

    # PRINT THE LENGTHS OF DATALOASER
    print("Length of train_loader", len(train_loader))
    print("Length of val_loader", len(val_loader))
    print("Length of test_loader", len(test_loader))
    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    modified_resnet = modify_resnet_model(encoder)
    capsule_network = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="SR").to(args.device)
    
    # initialize model
    model = SimCLR(modified_resnet, capsule_network)
     
    device = args.device
    checkpoint = torch.load("pre_trained_model_epoch_1000.pth", map_location=device)  
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(args.device)
    # print(model) 
    print("Model loaded from pre-trained model successfully")
    for param in model.parameters():
        param.requires_grad = False
 
    for param in model.caps_net.fc.parameters():
        param.requires_grad = True
     
    # check again if frozen
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
       
    model = model.to(args.device)
    
    start_epoch = 0
    criterion = nn.NLLLoss().to(args.device)
    optimizer = optim.SGD(model.caps_net.fc.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-6)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.logistic_epochs)
    num_train = len(train_loader.dataset)
    num_valid = len(val_loader.dataset)
    num_test = len(test_loader.dataset)
    best_valid_acc = 0
    for epoch in range(args.logistic_epochs):
        losses = AverageMeter()
        accs = AverageMeter()
        model.eval()
        model.caps_net.fc.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device) 
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

            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(args.device), y.to(args.device)

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
        # run["after_pretraining/validatin/epoch/acc"].log(valid_acc)  
        
        
        # decay lr
        scheduler.step()
      
    save_checkpoint(
            {'epoch': epoch + 1,
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
    
    # run["after_pretraining/testing/epoch/loss"].log(error)
    # run["after_pretraining/testing/epoch/acc"].log(perc)
    
    # run.stop()     