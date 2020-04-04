import numpy as np
import random
import datetime
import glob
from tqdm import tqdm

# Pytorch import
import torch
import torch.optim as optim                        # Import optimizer module from pytorch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# Tensorflow import
import tensorflow as tf

# Matplotlib import
import matplotlib.pyplot as plt

from .models.model_s10 import ResNet18
# from .data_manager.data_manager_pytorch import DataManager
from .data_manager.data_manager_albumentations import DataManager
from .training import Train
from .testing import Test

__assignment_name__ = 's10'

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

        # Holder for logs
        self.summary = {}

    

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
    
    def load_summary(self, index=-1):
        """
        This function loads summary after training
        """
        train_log_file = sorted(glob.glob(f'{self.train_dir_suffix}_*/events.out.tfevents.*'))[index]
        test_log_file = sorted(glob.glob(f'{self.test_dir_suffix}_*/events.out.tfevents.*'))[index]

        experiment_data = {self.name: (train_log_file, test_log_file)}
        self.summary = {}
        for experiment, (train, test) in experiment_data.items():
            train_data = {}
            test_data = {}
            self.summary[experiment] = {}
            for e in tf.compat.v1.train.summary_iterator(train):
                for v in e.summary.value:
                    if v.tag not in train_data:
                        train_data[v.tag] = []
                    train_data[v.tag].append(v.simple_value)
            for e in tf.compat.v1.train.summary_iterator(test):
                    for v in e.summary.value:
                        if v.tag not in test_data:
                            test_data[v.tag] = []
                        test_data[v.tag].append(v.simple_value)
            self.summary[experiment]['train'] = train_data
            self.summary[experiment]['test'] = test_data
    
    def plot_metric(self, metric='accuracy', figsize=(15, 5), ylim=[40, 100]):
        """
        This function loads a specific metric after training
        """
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.plot(self.summary[self.name]['train'][metric])
        axs.plot(self.summary[self.name]['test'][metric])
        axs.set_title(f'Loss plot {self.name}'.upper())
        axs.set_ylabel(metric.upper())
        axs.set_xlabel('EPOCH')
        axs.set_ylim(ylim)
        axs.legend(['train', 'test'], loc='best')

        plt.show()
    
    def get_mis_classified(self, no_of_images=25):
        """
        This function gets the misclassified images
        """
        test_loader_iterator = iter(self.data_manager.test_loader)
        fail_count = 0
        failed_samples = []
        while fail_count < no_of_images:
            data, target = test_loader_iterator.next()
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)

            pred = output.argmax(dim=1, keepdim=True)
            failed_index = ~pred.eq(target.view_as(pred)).squeeze()

            failed_data = data[failed_index]
            failed_target = target[failed_index]
            failed_prediction = pred[failed_index]
            
            batch_fail_count = failed_data.size(dim=0)
            fail_count += batch_fail_count

            for count in range(batch_fail_count) and fail_count <= no_of_images:
                failed_sample = {
                    'data': failed_data[count],
                    'target': failed_target[count],
                    'prediction': failed_prediction[count].item()
                }

                failed_samples.append(failed_sample)

        return failed_samples