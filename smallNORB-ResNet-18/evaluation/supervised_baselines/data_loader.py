import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Subset
from norb import smallNORB
from torch.utils.data import Dataset
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
    if dataset == "smallnorb":
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


DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
    'smallnorb': {'size': 32, 'channels': 1, 'classes': 5},
}

VIEWPOINT_EXPS = ['azimuth', 'elevation']
