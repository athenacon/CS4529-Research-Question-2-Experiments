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
    

def create_class_balanced_split(dataset, val=0.1):
    
    number_of_stl_classes = 10
    total_samples_per_class = len(dataset) // number_of_stl_classes   
    samples_per_class = {i: [] for i in range(number_of_stl_classes)}
     
    for idx, (_, label) in enumerate(dataset):
        samples_per_class[label].append(idx)
    
    train_indices = []
    valid_indices = []
     
    num_validation_samples_per_class = int(total_samples_per_class * val)
     
    for _, indices in samples_per_class.items():
        np.random.shuffle(indices)
        valid_indices.extend(indices[:num_validation_samples_per_class])
        train_indices.extend(indices[num_validation_samples_per_class:])
     
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    
    return train_subset, valid_subset
normalize = transforms.Normalize(mean=[0.4914, 0.4823,0.4466],
                                        std=[0.247, 0.243, 0.261]) 
    
def get_train_valid_loader( 
                           batch_size,
                           workers=4,
                           ):


    train_transform = transforms.Compose(
        [   transforms.RandomResizedCrop((96, 96), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
         ])
    
    val_test_transform = transforms.Compose(
        [
            transforms.CenterCrop((96 * 0.875, 96 * 0.875)),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            normalize
        ]
    )
    from torchvision.datasets import STL10
    
    full_dataset = STL10(root="stl10", split="train", download=True, transform=None)

    # call the function to create the datasets
    train_dataset, val_dataset = create_class_balanced_split(full_dataset)

            
    train_dataset = ApplyTransform(train_dataset, transform=train_transform)
    val_dataset = ApplyTransform(val_dataset, transform=val_test_transform) 
    # import torch
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
 
     
    print("Train dataset length: ", len(train_loader.dataset))
    print("Valid dataset length: ", len(valid_loader.dataset)) 
     
      
    
    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset) 

    print("\n[*] Train on {} samples, validate on {} samples".format(
        num_train, num_valid)
    )
    
    return train_loader, valid_loader 


def get_test_loader( 
                    batch_size, 
                    workers=4):
    val_test_transform = transforms.Compose(
        [
            transforms.CenterCrop((96 * 0.875, 96 * 0.875)),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            normalize
        ]
    )
    from torchvision.datasets import STL10
    from torch.utils.data import DataLoader
    test_dataset = STL10(root="stl10", split="test", transform=val_test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return test_loader


DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
    'smallnorb': {'size': 32, 'channels': 1, 'classes': 5},
    'stl10': {'size': 96, 'channels': 3, 'classes': 10},
}

VIEWPOINT_EXPS = ['azimuth', 'elevation']
