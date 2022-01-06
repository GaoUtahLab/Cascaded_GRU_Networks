# coding: utf-8

import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_waveform import EMDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Load data
###############################################################################
cwd = os.getcwd()

filename = 'dataset_waveform.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset'





test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=47, shuffle=False)


###############################################################################
# Main Cascased GRU model strucuture. It contains three following class:
#'Encoder_RNN': the encoder GRU structure
#'Decoder_RNN': the defcoder GRU strucutre
#'Seq2seq': the wrapper class contains both Encoder_RNN and Decoder_RNN to construct one cascaded GRU newtwork cell.
###############################################################################
class Encoder_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(Encoder_RNN, self).__init__()
        self.batchnorm = nn.BatchNorm1d(1,500)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
        )

    def forward(self, x):
        r_out, h_n = self.rnn(x)
        return r_out, h_n


class Decoder_RNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layer):
        super(Decoder_RNN, self).__init__()
        self.rnn_de = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layer,
            # bidirectional=True,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, hn):
        r_out, h_n = self.rnn_de(x,hn)
        out = self.out(r_out)
        return out


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder



    def forward(self, x, target, batch_size):
        total_seq = target.shape[0]
        r_out, h_n = self.encoder(x)
        out = self.decoder(x, h_n)
        return out, h_n

def evaluate(modelw,modelw1,modelw2):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :175]
            targetw = ori_data[:, :, 175:1000]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[50:,:,:], targetw, batch_size)
            w_out2, hidden = modelw1(torch.cat((data[50:,:,:],w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data[100:,:,:],w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-450:,:,:], targetw[-450:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss, hidden



#load model
if device =='cuda':
    modelw2 = torch.load('GRU_model_same_seq_125_250.pkl')
    modelw1 = torch.load('GRU_model_same_seq_250_500.pkl')
    modelw = torch.load('GRU_model_same_seq_500_950.pkl')
else:
    modelw2 = torch.load('GRU_model_same_seq_125_250.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_same_seq_250_500.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_same_seq_500_950.pkl', map_location=torch.device('cpu'))



criterion = nn.MSELoss()
modelw.to(device)
modelw1.to(device)
modelw2.to(device)
# model_c.to(device)

data, targetw, predictw, loss, hidden = evaluate(modelw,modelw1,modelw2)
data = data.permute(1,2,0)
predictw = predictw.permute(1,2,0)
targetw = targetw.permute(1,2,0)
predictw = predictw.cpu().detach().numpy()
targetw = targetw.cpu().detach().numpy()

data = data.cpu().detach().numpy()

vislat = np.array(hidden.cpu().detach().numpy())
vislat = vislat.astype(np.float64)
vislat3 = vislat[3,:,:]


tsne3 = manifold.TSNE(n_components=2,perplexity=5,learning_rate = 100)
X_tsne3 = tsne3.fit_transform(vislat3)


plt.scatter(X_tsne3[:21,0],X_tsne3[:21,1],label='0T - 1.5T', c='y', alpha=0.5)
plt.scatter(X_tsne3[21:35,0],X_tsne3[21:35,1],label='1.6T - 2.6T', c='b', alpha=0.5)
plt.scatter(X_tsne3[35:,0],X_tsne3[35:,1],label='2.7T - 4.5T', c='r', alpha=0.5)

plt.legend()
plt.savefig('hidden_state_visulization.pdf')
plt.close()

print('visulization demo done')
