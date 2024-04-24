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
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from utils import yaml_config_hook
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn, optim

from simclr.modules.resnet_hacks import modify_resnet_model
# Capsule Network
from capsule_network import resnet20
from utilscapsnet import AverageMeter
def create_class_balanced_split(dataset, num_samples_per_class=500): 
    
    number_of_cifar_classes = 10   
    samples_per_class = {i: [] for i in range(number_of_cifar_classes)}  
    for idx, (_, label) in enumerate(dataset):
        samples_per_class[label].append(idx)
    
    train_indices = []
    valid_indices = []
    
    # For each class, randomly choose n_samples_per_class for both subsets
    for _, indices in samples_per_class.items():
        np.random.shuffle(indices)
        train_indices.extend(indices[:num_samples_per_class])
        valid_indices.extend(indices[num_samples_per_class:2*num_samples_per_class])
    
    # Create subset datasetss
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    
    return train_subset, valid_subset

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
    filename = "simclr_finetune_1" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
        
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")
    finetune_1_percent_n_epochs = 20
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
    full_dataset = CIFAR10(root=args.dataset_dir, train=True, download=True, transform=None)

    # call the function to create the datasets
    train_dataset, val_dataset = create_class_balanced_split(full_dataset)

            
    train_dataset = ApplyTransform(train_dataset, transform=train_transform)
    val_dataset = ApplyTransform(val_dataset, transform=test_transform) 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
 
    test_dataset = CIFAR10(root=args.dataset_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    
    print("Train dataset length: ", len(train_loader.dataset))
    print("Valid dataset length: ", len(valid_loader.dataset))
    print("Test dataset length: ", len(test_loader.dataset))
     
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
    print(model) 
    print("Model loaded from pre-trained model successfully")
    for param in model.parameters():
        param.requires_grad = True
       
    model = model.to(args.device)
    
    start_epoch = 0
    criterion = nn.NLLLoss().to(args.device)
     
    all = [p for p in model.parameters()]
    print(f"Total model parameters: {len(all)}")
    
    caps_net_fc_params = list(model.caps_net.fc.parameters())
    other_params = [p for p in model.parameters() if not any(p is pp for pp in caps_net_fc_params)]

    # sanity check
    print(f"Total model parameters: {len(list(model.parameters()))}")
    print(f"CapsNet FC parameters: {len(caps_net_fc_params)}")
    print(f"Other parameters: {len(other_params)}")

    param_groups = [
        {'params': other_params, 'lr': 0.05},
        {'params': caps_net_fc_params, 'lr': 0.01}
    ]

    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_1_percent_n_epochs)

    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)
    num_test = len(test_loader.dataset)
    best_valid_acc = 0
    for epoch in range(finetune_1_percent_n_epochs):
        losses = AverageMeter()
        accs = AverageMeter()
        model.train() 
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

            for i, (x, y) in enumerate(valid_loader):
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