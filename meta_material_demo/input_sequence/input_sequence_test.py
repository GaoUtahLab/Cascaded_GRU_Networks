# coding: utf-8
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_test import EMDataset
import matplotlib.pyplot as plt
from util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Load data
###############################################################################
cwd = os.getcwd()
filename = 'Ex_data_average_final_set.mat'   ##load the .mat file to the current directory and change the name here
HV_input1 = 'Ex_test_a'                         ##variable name for the HV in the .mat.file
dir = os.path.join(cwd,filename)


test = EMDataset(dir, HV_input1, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=1000, shuffle=False)

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
        return out




#load model
if device =='cuda':
    model = torch.load('GRU_model_same_seq_25_50_final.pkl')
    model00 = torch.load('GRU_model_same_seq_50_100_final.pkl')
    model0 = torch.load('GRU_model_same_seq_100_200_final.pkl')
    model1 = torch.load('GRU_model_same_seq_200_400_final.pkl')
    model2 = torch.load('GRU_model_same_seq_400_600_final.pkl')
    model3 = torch.load('GRU_model_same_seq_600_800_final.pkl')
    model4 = torch.load('GRU_model_same_seq_800_1200_final.pkl')
    model5 = torch.load('GRU_model_same_seq_1200_1600_final.pkl')

else:
    model = torch.load('GRU_model_same_seq_25_50_final.pkl',map_location=torch.device('cpu'))
    model00 = torch.load('GRU_model_same_seq_50_100_final.pkl',map_location=torch.device('cpu'))
    model0 = torch.load('GRU_model_same_seq_100_200_final.pkl',map_location=torch.device('cpu'))
    model1 = torch.load('GRU_model_same_seq_200_400_final.pkl',map_location=torch.device('cpu'))
    model2 = torch.load('GRU_model_same_seq_400_600_final.pkl',map_location=torch.device('cpu'))
    model3 = torch.load('GRU_model_same_seq_600_800_final.pkl',map_location=torch.device('cpu'))
    model4 = torch.load('GRU_model_same_seq_800_1200_final.pkl',map_location=torch.device('cpu'))
    model5 = torch.load('GRU_model_same_seq_1200_1600_final.pkl',map_location=torch.device('cpu'))

model.to(device)
model00.to(device)
model0.to(device)
model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)
model5.to(device)

criterion = nn.MSELoss()

# result with 25 input sequence
data, predict, target, total_data, total_loss = evaluate_25(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'input_seq_25_result'
fig_gen(predict, target, filename)
print('input sequence = 25|, MSE Loss: {:5.6f}'.format(total_loss))

# result with 50 input sequence
data, predict, target, total_data, total_loss = evaluate_50(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'input_seq_50_result'
fig_gen(predict, target, filename)
print('input sequence = 50|, MSE Loss: {:5.6f}'.format(total_loss))


# result with 100 input sequence
data, predict, target, total_data, total_loss = evaluate_100(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'input_seq_100_result'
fig_gen(predict, target, filename)
print('input sequence = 100|, MSE Loss: {:5.6f}'.format(total_loss))


# result with 200 input sequence
data, predict, target, total_data, total_loss = evaluate_200(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'input_seq_200_result'
fig_gen(predict, target, filename)
print('input sequence = 200|, MSE Loss: {:5.6f}'.format(total_loss))


# result with 400 input sequence
data, predict, target, total_data, total_loss = evaluate_400(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'input_seq_400_result'
fig_gen(predict, target, filename)
print('input sequence = 400|, MSE Loss: {:5.6f}'.format(total_loss))


# result with 600 input sequence
data, predict, target, total_data, total_loss = evaluate_600(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'input_seq_600_result'
fig_gen(predict, target, filename)
print('input sequence = 600|, MSE Loss: {:5.6f}'.format(total_loss))


# result with 800 input sequence
data, predict, target, total_data, total_loss = evaluate_800(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
filename = 'input_seq_800_result'
fig_gen(predict, target, filename)
print('input sequence = 800|, MSE Loss: {:5.6f}'.format(total_loss))


print('input sequence demo finished')
