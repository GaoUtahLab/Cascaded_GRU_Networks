# coding: utf-8
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_waveform_noise import EMDataset
import matplotlib.pyplot as plt
import scipy.io as sio



device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Load data
###############################################################################
cwd = os.getcwd()
filename = 'dataset_waveform.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset'
HV_input22 = 'testlabel'




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



def evaluate(modelw,modelw1,modelw2, test_loader):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 50:175]
            targetw = ori_data[:, :, 175:1000]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data, targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data,w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data[50:,:,:],w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-450:,:,:], targetw[-450:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss


def fig_gen(predictw, targetw, filename):
    #plot generate
    predictw = predictw.permute(1, 2, 0)
    targetw = targetw.permute(1, 2, 0)
    predictw = predictw.cpu().detach().numpy()
    targetw = targetw.cpu().detach().numpy()
    plt.plot(predictw[5, 0, :], marker='o', markerfacecolor='red', markersize=2, color='blue', linewidth=2,
             label="predicted")
    plt.plot(targetw[5, 0, :], marker='*', markerfacecolor='yellow', markersize=2, color='purple', linewidth=2,
             label="actual")
    plt.legend()
    plt.savefig(filename+'.pdf')
    plt.close()

    #raw data save (as .mat)
    my_dictonary = {}
    newkey = 'predicted data'
    my_dictonary[newkey] = predictw
    newkey = 'target data'
    my_dictonary[newkey] = targetw
    sio.savemat(filename+'.mat', my_dictonary)
    return


def noise_test(A):

    test = EMDataset(dir, HV_input2, A, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test, batch_size=50, shuffle=False)

    data, targetw, predictw, loss = evaluate(modelw, modelw1, modelw2, test_loader)
    filename = 'A_'+ str(A) + 'sample_result'
    fig_gen(predictw, targetw, filename)

    print('A = {:5.2f}|, MSE Loss: {:5.6f}'.format(A, loss))


#load model
if device =='cuda':
    modelw2 = torch.load('GRU_model_same_seq_125_250.pkl')
    modelw1 = torch.load('GRU_model_same_seq_250_500.pkl')
    modelw = torch.load('GRU_model_same_seq_500_950.pkl')
else:
    modelw2 = torch.load('GRU_model_same_seq_125_250.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_same_seq_250_500.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_same_seq_500_950.pkl', map_location=torch.device('cpu'))





modelw.to(device)
modelw1.to(device)
modelw2.to(device)

criterion = nn.MSELoss()
A = 0.01
noise_test(A)
A = 0.02
noise_test(A)
A = 0.05
noise_test(A)
A = 0.07
noise_test(A)
A = 0.1
noise_test(A)
A = 0.15
noise_test(A)
A = 0.2
noise_test(A)


print('noise demo finished')
