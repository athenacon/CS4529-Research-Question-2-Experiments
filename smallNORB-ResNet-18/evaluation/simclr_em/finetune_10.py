import os
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import neptune
from simclr import SimCLR
from loss import EmRoutingLoss

from utils import yaml_config_hook
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn, optim

from simclr.modules.resnet_hacks import modify_resnet_model
# Capsule Network
from capsule_network import resnet20
from utilscapsnet import AverageMeter
from norb import smallNORB
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

from torch.utils.data import DataLoader

def get_balanced_train_valid_loaders(data_dir, dataset, batch_size, num_workers=4, pin_memory=False):
    
    data_dir = data_dir + '/' + dataset 
    dataset = smallNORB(data_dir, train=True, download=True, transform=None)
    
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
    # Organize indices by class
    class_indices = {i: [] for i in range(5)}  #  5 classes
    for idx, (_, target) in enumerate(dataset):
        class_indices[int(target)].append(idx)

    n_samples_per_class = len(dataset) * 0.1 // 5
    n_samples_per_class = int(n_samples_per_class)
    
    print("n_samples_per_class", n_samples_per_class)
   
    train_idx = []
    valid_idx = [] 
        
    for _, indices in class_indices.items():
        np.random.seed(seed_value)
        np.random.shuffle(indices)
        class_train_indices = indices[:n_samples_per_class]
        
        class_valid_indices = indices[n_samples_per_class:n_samples_per_class * 2]

        train_idx.extend(class_train_indices)
        valid_idx.extend(class_valid_indices)
 
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)

    train_set = ApplyTransform(train_set, transform=transforms.Compose(trans_train))
    valid_set = ApplyTransform(valid_set, transform=transforms.Compose(trans_valid))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

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
              
    
    kwargs = {'num_workers': 0, 'pin_memory': False}
    finetune_1_percent_n_epochs = 20


    data_loader = get_balanced_train_valid_loaders(
        "./data", "smallNorb", args.batch_size,  
       args.workers, False
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
    
    # initialize ResNet
    from simclr.modules.resnet_hacks import modify_resnet_model
    from simclr.modules import get_resnet

    encoder = get_resnet(args.resnet, pretrained=False)
    modified_resnet = modify_resnet_model(encoder)
    
    # encoder = resnet.resnet20()
    capsule_network = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="EM").to(args.device)
    
    # initialize model
    model = SimCLR(encoder, capsule_network)
    
    device = args.device
    checkpoint = torch.load("pre_trained_model_epoch_1000.pth", map_location=device)  
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(args.device)
     
    print("Model loaded from pre-trained model successfully")
    for param in model.parameters():
        param.requires_grad = True 
       
    model = model.to(args.device)
    
    start_epoch = 0
    criterion = EmRoutingLoss(finetune_1_percent_n_epochs).to(args.device)

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
        model.train() 
        losses = AverageMeter()
        accs = AverageMeter()
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device) 
            out = model(x)
            
            loss = criterion(out, y, epoch = epoch)

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
                
                loss = criterion(out, y, epoch = epoch)

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