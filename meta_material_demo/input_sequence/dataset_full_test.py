# dataloader file for metamaterial application


from __future__ import print_function, division
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader




class EMDataset(Dataset):
    """dataset loader"""

    def __init__(self, dir, data_input, transform=None):
        """
        Args:
            dir (string): Path to the .mat file that contains the data.
            data_input (string): variable name of the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = loadmat(dir)
        self.input_data = self.x.get(data_input) * 1000
        self.input_data = self.input_data.reshape(1000,1,-1)
        self.input_data = self.input_data.astype(np.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self,index):
        HV = self.input_data[index,:,:]
        return HV
