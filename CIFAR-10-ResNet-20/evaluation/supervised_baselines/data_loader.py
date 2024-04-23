import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset, random_split
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

    if dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.446],
            std=[0.247, 0.243, 0.261]
            )
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
    )
        
        training_dataset_size = 45000
        val_dataset_size = 5000
        
        train_dataset, val_dataset = random_split(full_training_dataset, [training_dataset_size, val_dataset_size])
        
        train_dataset = ApplyTransform(train_dataset, transform=train_transform)
        val_dataset = ApplyTransform(val_dataset, transform=test_transform)
        kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **kwargs
        )
        valid_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **kwargs)
            
    return train_loader, valid_loader

def get_test_loader(data_dir,
                    dataset,
                    batch_size,
                    exp='azimuth', # smallnorb only
                    familiar=True, # smallnorb only
                    num_workers=4,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201))]
        
        dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                transform=transforms.Compose(trans))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10}
}