# coding: utf-8
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_waveform_sampling import EMDataset
import matplotlib.pyplot as plt
import scipy.io as sio
from util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cwd = os.getcwd()



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
        out = torch.zeros(total_seq, batch_size, 1).to(device)
        r_out, h_n = self.encoder(x)
        out = self.decoder(x, h_n)

        return out, h_n



criterion = nn.MSELoss()

# load model w sampling = 0.5
filename = 'dataset_wave_sampling.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset_half'
test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=50, shuffle=False)


if device =='cuda':
    modelw2 = torch.load('GRU_model_sampling_0.5_250_500.pkl')
    modelw1 = torch.load('GRU_model_sampling_0.5_500_1000.pkl')
    modelw = torch.load('GRU_model_sampling_0.5_1000_2000.pkl')
else:
    modelw2 = torch.load('GRU_model_sampling_0.5_250_500.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_sampling_0.5_500_1000.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_sampling_0.5_1000_2000.pkl', map_location=torch.device('cpu'))


modelw.to(device)
modelw1.to(device)
modelw2.to(device)


data, targetw, predictw, loss = evaluate_05(modelw,modelw1,modelw2, test_loader, criterion)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'sampling_0.5_result'
fig_gen(predictw, targetw, filename)

print('demo finished with sampling = 0.5')


# load model w sampling = 2
filename = 'dataset_wave_sampling.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset_s2'
test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=50, shuffle=False)

if device =='cuda':
    modelw2 = torch.load('GRU_model_sampling_2_75_125.pkl')
    modelw1 = torch.load('GRU_model_sampling_2_125_250.pkl')
    modelw = torch.load('GRU_model_sampling_2_250_500.pkl')
else:
    modelw2 = torch.load('GRU_model_sampling_2_75_125.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_sampling_2_125_250.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_sampling_2_250_500.pkl', map_location=torch.device('cpu'))

modelw.to(device)
modelw1.to(device)
modelw2.to(device)


data, targetw, predictw, loss = evaluate_2(modelw,modelw1,modelw2, test_loader, criterion)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'sampling_2_result'
fig_gen(predictw, targetw, filename)

print('demo finished with sampling = 2')


# load model w sampling = 3
filename = 'dataset_wave_sampling.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset_s3'
test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=50, shuffle=False)

if device =='cuda':
    modelw2 = torch.load('GRU_model_sampling_3_45_85.pkl')
    modelw1 = torch.load('GRU_model_sampling_3_85_165.pkl')
    modelw = torch.load('GRU_model_sampling_3_165_330.pkl')
else:
    modelw2 = torch.load('GRU_model_sampling_3_45_85.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_sampling_3_85_165.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_sampling_3_165_330.pkl', map_location=torch.device('cpu'))

modelw.to(device)
modelw1.to(device)
modelw2.to(device)

data, targetw, predictw, loss = evaluate_3(modelw,modelw1,modelw2, test_loader, criterion)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'sampling_3_result'
fig_gen(predictw, targetw, filename)

print('demo finished with sampling = 3')


# load model w sampling = 4
filename = 'dataset_wave_sampling.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset_s4'
test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=50, shuffle=False)

if device =='cuda':
    modelw2 = torch.load('GRU_model_sampling_4_35_65.pkl')
    modelw1 = torch.load('GRU_model_sampling_4_65_125.pkl')
    modelw = torch.load('GRU_model_sampling_4_125_250.pkl')
else:
    modelw2 = torch.load('GRU_model_sampling_4_35_65.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_sampling_4_65_125.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_sampling_4_125_250.pkl', map_location=torch.device('cpu'))

modelw.to(device)
modelw1.to(device)
modelw2.to(device)

data, targetw, predictw, loss = evaluate_4(modelw,modelw1,modelw2, test_loader, criterion)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'sampling_4_result'
fig_gen(predictw, targetw, filename)

print('demo finished with sampling = 4')


# load model w sampling = 5
filename = 'dataset_wave_sampling.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset_s5'
test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=50, shuffle=False)

if device =='cuda':
    modelw2 = torch.load('GRU_model_sampling_5_25_50.pkl')
    modelw1 = torch.load('GRU_model_sampling_5_50_100.pkl')
    modelw = torch.load('GRU_model_sampling_5_100_200.pkl')
else:
    modelw2 = torch.load('GRU_model_sampling_5_25_50.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_sampling_5_50_100.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_sampling_5_100_200.pkl', map_location=torch.device('cpu'))

modelw.to(device)
modelw1.to(device)
modelw2.to(device)

data, targetw, predictw, loss = evaluate_5(modelw,modelw1,modelw2, test_loader, criterion)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'sampling_5_result'
fig_gen(predictw, targetw, filename)

print('demo finished with sampling = 5')


# load model w sampling = 10
filename = 'dataset_wave_sampling.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset_s10'
test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=50, shuffle=False)

if device =='cuda':
    modelw2 = torch.load('GRU_model_sampling_10_15_25.pkl')
    modelw1 = torch.load('GRU_model_sampling_10_25_50.pkl')
    modelw = torch.load('GRU_model_sampling_10_50_100.pkl')
else:
    modelw2 = torch.load('GRU_model_sampling_10_15_25.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_sampling_10_25_50.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_sampling_10_50_100.pkl', map_location=torch.device('cpu'))

modelw.to(device)
modelw1.to(device)
modelw2.to(device)


data, targetw, predictw, loss = evaluate_10(modelw,modelw1,modelw2, test_loader, criterion)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'sampling_10_result'
fig_gen(predictw, targetw, filename)

print('demo finished with sampling = 10')


# load model w sampling = 20
filename = 'dataset_wave_sampling.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset_s20'
test = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=50, shuffle=False)

if device =='cuda':
    modelw2 = torch.load('GRU_model_sampling_20_10_15.pkl')
    modelw1 = torch.load('GRU_model_sampling_20_15_25.pkl')
    modelw = torch.load('GRU_model_sampling_20_25_50.pkl')
else:
    modelw2 = torch.load('GRU_model_sampling_20_10_15.pkl', map_location=torch.device('cpu'))
    modelw1 = torch.load('GRU_model_sampling_20_15_25.pkl', map_location=torch.device('cpu'))
    modelw = torch.load('GRU_model_sampling_20_25_50.pkl', map_location=torch.device('cpu'))

modelw.to(device)
modelw1.to(device)
modelw2.to(device)

data, targetw, predictw, loss = evaluate_20(modelw,modelw1,modelw2, test_loader, criterion)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'sampling_20_result'
fig_gen(predictw, targetw, filename)

print('demo finished with sampling = 20')
