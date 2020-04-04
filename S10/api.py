import numpy as np
import random
import datetime
from tqdm import tqdm

import torch
import torch.optim as optim                        # Import optimizer module from pytorch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from .models.model_s9 import ResNet18
# from .data_manager.data_manager_pytorch import DataManager
from .data_manager.data_manager_albumentations import DataManager
from .training import Train
from .testing import Test

__assignment_name__ = 's9'

class Experiment(object):
    """
    Experiment class provides interface to all steps invloved in training CNNs
    """
    def initialize(self):
        """
        This function checks for prerequisites and sets seeds for reproducability
        """
        seed = 1
        
        # Set python random seed
        random.seed(seed)

        # Set numpy seed
        np.random.seed(seed) 

        # Set pytorch seed
        use_cuda = torch.cuda.is_available()
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)

        print(f'CUDA status: {use_cuda}')

        return use_cuda

    def __init__(self, name, dataset_name='cifar10'):
        super().__init__()
        self.name = name
        self.dataset_name = dataset_name
        self.use_cuda = self.initialize()
        self.device = torch.device("cuda" if self.use_cuda else "cpu") # Initializing GPU
        
        # Initializing directories to save log
        self.dir_suffix = f'/content/drive/My Drive/log_{self.dataset_name}_{__assignment_name__}'
        self.train_dir_suffix = f'{self.dir_suffix}/run_train_{self.name}'
        self.test_dir_suffix = f'{self.dir_suffix}/run_test_{self.name}'

        # Initializing model
        self.model = ResNet18().to(device=self.device)

        # Initializing data
        self.data_manager = DataManager(dataset_name=dataset_name)

    

    def run(self, epochs=40, momentum=0.9, lr=0.01, regularization=None, weight_decay=0.01):
        """
        THis function runs the experiment
        """
        if hasattr(tqdm, '_instances'):
            tqdm._instances.clear()

        now = datetime.datetime.now()
        prefix = now.strftime('%m-%d-%y %H:%M:%S')

        train_dir = f'{self.train_dir_suffix}_{prefix}'
        test_dir = f'{self.test_dir_suffix}_{prefix}'

        train_writer = SummaryWriter(train_dir)
        test_writer = SummaryWriter(test_dir)

        if regularization == 'L2' or regularization == 'L1 and L2':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

        train = Train(model=self.model, optimizer=optimizer, device=self.device, train_loader=self.data_manager.train_loader, writer=train_writer)
        test = Test(model=self.model, device=self.device, test_loader=self.data_manager.test_loader, writer=test_writer)
        
        for epoch in range(0, epochs):
            train.step(epoch, regularization, weight_decay)
            test.step(epoch, regularization, weight_decay)
            scheduler.step()