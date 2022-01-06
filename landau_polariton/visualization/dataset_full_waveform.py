# dataloader file for Landau Prolariton application
# Yingheng Tang

from __future__ import print_function, division
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader





class EMDataset(Dataset):
    """dataset loader"""

    def __init__(self, dir, data_input, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = loadmat(dir)
        self.input_data = self.x.get(data_input) * 1000000
        self.input_data = self.input_data.reshape(-1,1,1000)
        self.input_data = self.input_data.astype(np.float32)


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self,index):
        HV = self.input_data[index,:,:]
        return HV
