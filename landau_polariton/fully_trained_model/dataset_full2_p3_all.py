# dataloader file for Landau Prolariton
# Yingheng Tang

from __future__ import print_function, division

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader




class EMDataset(Dataset):
    """dataset loader"""

    def __init__(self, dir, data_input, data_input2, transform=None):
        """
        Args:
            dir (string): Path to the .mat file that contains the data.
            data_input (string): variable name of the data.
            data_input2 (string): variable name of the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = loadmat(dir)
        self.input_data = self.x.get(data_input) * 1000000
        self.input_data = self.input_data.reshape(-1,1,1000)
        self.input_data = self.input_data.astype(np.float32)

        self.input_target = self.x.get(data_input2)
        self.input_target = self.input_target.astype(np.float32)

        self.input_target[:,0] = self.input_target[:,0] * 1000 - 430
        self.input_target[:,1] = self.input_target[:,1] * 1000 - 340
        self.input_target[:,2] = self.input_target[:,2] * 1000 - 630






    def __len__(self):
        return len(self.input_data)

    def __getitem__(self,index):
        HV = self.input_data[index,:,:]
        TG = self.input_target[index,:]


        return HV, TG
