import torch
import numpy as np
import random

class Experiment(object):
    """
    Experiment class provides interface to all steps invloved in training CNNs
    """

    @staticmethod
    def initialize():
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

        return {
            'use_cuda': use_cuda
        }

    def __init__(self):
        super().__init__()