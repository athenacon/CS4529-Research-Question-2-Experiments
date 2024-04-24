import torchvision
class TrainTransform(object):
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x Ìƒi and x Ìƒj, which we consider as a positive pair.
    """

    def __init__(self):
        from torchvision import transforms
        normalize = transforms.Normalize(mean=[0.4914, 0.4823, 0.4466],
                                        std=[0.247, 0.243, 0.261]) 
        from torchvision.transforms import InterpolationMode
        self.train_transform = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop((96, 96), scale=(0.2, 1.0),  interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize   
            ] 
        )

         
        self.train_transform_prime = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop((96, 96), scale=(0.2, 1.0),  interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize   
            ] 
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform_prime(x)
