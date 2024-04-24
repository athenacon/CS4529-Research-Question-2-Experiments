import torchvision

class TrainTransform(object):
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self):
        # All transformations are from the SimCLR paper
        # strength is 0.5 for CIFAR-10 per the SimCLR paper
        s = 0.5
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=32),
                torchvision.transforms.RandomHorizontalFlip(), # with 0.5 probability by default
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                                 std=[0.247, 0.243, 0.261]),
            ]
        )
        
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)