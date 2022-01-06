# coding: utf-8
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_p3_test_final import EMDataset
import matplotlib.pyplot as plt
import scipy.io as sio


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Load data
###############################################################################
cwd = os.getcwd()

filename = 'dataset_app3_new_5_diff.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)                       ##variable name for the HV in the .mat.file
HV_input1 = 'test_m5'
dir = os.path.join(cwd,filename)

test = EMDataset(dir, HV_input1, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=200, shuffle=False)

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



def evaluate_40(model0, model1, model2, model3):
    # Turn on evaluation mode which disables dropout.
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, ori_data in enumerate(test_loader):
            data = ori_data[:, :, :40]
            targets = ori_data[:, :, 40:]
            data = data.to(device)
            targets = targets.to(device)
            data = data.permute(2, 0, 1)
            data_input1 = data[5:40,:,:]
            batch_size = targets.shape[1]
            targets = targets.permute(2, 0, 1)
            output1 = model3(data_input1, targets, batch_size)
            output2 = model2(torch.cat((data, output1), dim=0), targets, batch_size)
            output3 = model1(torch.cat((data, output1, output2), dim=0) ,targets,batch_size)
            data2 = torch.cat((data, output1, output2, output3), dim=0)
            output4 = model0(data2,targets,batch_size)
            output = torch.cat((output1, output2, output3, output4), dim=0)
            total_loss = criterion(output[-300:,:,:], targets[-300:,:,:])

    return data, output, targets, total_loss.item()


def evaluate_75(model0, model1, model2):
    # Turn on evaluation mode which disables dropout.
    model0.eval()
    model1.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, ori_data in enumerate(test_loader):
            data = ori_data[:, :, :75]
            targets = ori_data[:, :, 75:]
            data = data.to(device)
            targets = targets.to(device)
            data = data.permute(2, 0, 1)
            data_input1 = data
            batch_size = targets.shape[1]
            targets = targets.permute(2, 0, 1)
            output1 = model2(data_input1, targets, batch_size)
            output2 = model1(torch.cat((data_input1, output1), dim=0) ,targets,batch_size)
            data2 = torch.cat((data_input1, output1, output2), dim=0)
            output3 = model0(data2,targets,batch_size)
            output = torch.cat((output1, output2, output3), dim=0)
            total_loss = criterion(output[-300:,:,:], targets[-300:,:,:])
    return data, output, targets, total_loss.item()


#load model

if device =='cuda':
    model3 = torch.load('GRU_model_same_seq_40_normal_new_m5.pkl')
    model2 = torch.load('GRU_model_same_seq_75_normal_new_m5.pkl')
    model1 = torch.load('GRU_model_same_seq_150_normal_new_m5.pkl')
    model0 = torch.load('GRU_model_same_seq_300_normal_new_m5.pkl')
else:
    model3 = torch.load('GRU_model_same_seq_40_normal_new_m5.pkl',map_location=torch.device('cpu'))
    model2 = torch.load('GRU_model_same_seq_75_normal_new_m5.pkl',map_location=torch.device('cpu'))
    model1 = torch.load('GRU_model_same_seq_150_normal_new_m5.pkl',map_location=torch.device('cpu'))
    model0 = torch.load('GRU_model_same_seq_300_normal_new_m5.pkl',map_location=torch.device('cpu'))



model0.to(device)
model1.to(device)
model2.to(device)
model3.to(device)

criterion = nn.MSELoss()

#result wiht input seq of 40
data, predict, target, loss = evaluate_40(model0, model1, model2, model3)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
print('input sequence = 40|, MSE Loss: {:5.6f}'.format(loss))
filename = 'input_seq_40_result'
fig_gen(predict, target, filename)


#result wiht input seq of 75
data, predict, target, loss = evaluate_75(model0, model1, model2)
data = data.permute(1,2,0)
data = data.cpu().detach().numpy()
print('input sequence = 75|, MSE Loss: {:5.6f}'.format(loss))
filename = 'input_seq_75_result'
fig_gen(predict, target, filename)


print('input sequence demo finished')
