import torchvision
import torch
from torchvision import datasets, transforms # Import datasets and augmentation functionality from vision module within pytorch

class DataManager(object):
    """
    Class that handles data management for an experiment
    """
    
    def __init__(self, batch_size=64, use_cuda=True, dataset_name='cifar10'):
        super().__init__()

        self.dataset_name = dataset_name

        if self.dataset_name == 'cifar10':
            # Train Phase transformations
            self.train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                                # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                                                # transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

            # Test Phase transformations
            self.test_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
            
            self.trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.train_transforms)
            self.testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.test_transforms)

            # dataloader arguments - something you'll fetch these from cmdprmt
            dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=batch_size)

            # train dataloader
            self.train_loader = torch.utils.data.DataLoader(self.trainset, **dataloader_args)
            # test dataloader
            self.test_loader = torch.utils.data.DataLoader(self.testset, **dataloader_args)