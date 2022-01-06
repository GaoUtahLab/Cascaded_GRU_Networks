# coding: utf-8
import argparse
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full2_p3_all import EMDataset
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from scipy.fft import fft, fftfreq



device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Load data
###############################################################################
cwd = os.getcwd()
filename = 'dataset_waveform.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input2 = 'testset'
HV_input22 = 'testlabel_coe'




eval_batch_size = 100
test = EMDataset(dir, HV_input2, HV_input22, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=eval_batch_size, shuffle=False)


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


def evaluate(modelw1, modelw2, modelw3, modelc):
    # Turn on evaluation mode which disables dropout.
    modelw1.eval()
    modelw2.eval()
    modelw3.eval()
    modelc.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data, targets) in enumerate(test_loader):
            data = ori_data[:, :, 50:175]
            data = data.to(device)
            targetw = ori_data[:, :, 175:]

            targets = targets.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1, hidden1 = modelw1(data,targets,batch_size)
            data2 = torch.cat((data, w_out1), dim=0)

            w_out2, hidden2 = modelw2(data2, targets, batch_size)
            data3 = torch.cat((data2[50:, :, :], w_out2), dim=0)

            w_out3, hidden3 = modelw3(data3,targets,batch_size)
            w_out = torch.cat((w_out1, w_out2, w_out3), dim=0)


            # coe extraction
            output = modelc(data2, hidden2)
            output = output.permute(1, 2, 0)
            output = output[:, 0, -3:]


            loss = criterion(output, targets)
            total_loss += loss.item()

    return ori_data, targetw, w_out, targets, output



#load model
if device =='cuda':
    modelw1 = torch.load('GRU_model_same_seq_125_250.pkl')
    modelw2 = torch.load('GRU_model_same_seq_250_500.pkl')
    modelw3 = torch.load('GRU_model_same_seq_500_950.pkl')
    modelc = torch.load('GRU_model_same_seq_250_coe_fine_tune_all.pkl')

else:
    modelw1 = torch.load('GRU_model_same_seq_125_250.pkl', map_location=torch.device('cpu'))
    modelw2 = torch.load('GRU_model_same_seq_250_500.pkl', map_location=torch.device('cpu'))
    modelw3 = torch.load('GRU_model_same_seq_500_950.pkl', map_location=torch.device('cpu'))
    modelc = torch.load('GRU_model_same_seq_250_coe_fine_tune_all.pkl', map_location=torch.device('cpu'))




criterion = nn.MSELoss()
modelw1.to(device)
modelw2.to(device)
modelw3.to(device)
modelc.to(device)


#prediction and post process
data, targetw, predictw, targetc, predictc = evaluate(modelw1, modelw2, modelw3, modelc)
predictw = predictw.permute(1,2,0)
predictw = predictw.cpu().detach().numpy() / 1000000
targetw = targetw.cpu().detach().numpy() / 1000000
predictc = predictc.cpu().detach().numpy()
targetc = targetc.cpu().detach().numpy()
data = data.cpu().detach().numpy() / 1000000
predictc[:,0] = (predictc[:,0] + 430) / 1000
targetc[:,0] = (targetc[:,0] + 430) / 1000
predictc[:,1] = (predictc[:,1] + 340) / 1000
targetc[:,1] = (targetc[:,1] + 340) / 1000
predictc[:,2] = (predictc[:,2] + 630) / 1000
targetc[:,2] = (targetc[:,2] + 630) / 1000


#one sample result plot(time series)
plt.plot(predictw[5,0,:], marker='o', markerfacecolor='red', markersize=2, color='blue', linewidth=2,
             label="predicted")
plt.plot(targetw[5,0,:], marker='*', markerfacecolor='yellow', markersize=2, color='purple', linewidth=2,
             label="actual")
plt.legend()
plt.savefig('sample_raw_prediction_waveform.pdf')
plt.show()

#result plot(coeficient)
test = sio.loadmat('dataset_test.mat')
H_field = test['H_field_test']

plt.plot(predictc[:46,0], H_field[:46,0], marker='o', markerfacecolor='red', markersize=2, color='blue', linewidth=2,
             label="predicted_coe1")
plt.plot(targetc[:46,0], H_field[:46,0], marker='*', markerfacecolor='yellow', markersize=2, color='purple', linewidth=2,
             label="actual_coe1")
plt.plot(predictc[:39,1], H_field[:39,0], marker='o', markerfacecolor='red', markersize=2, color='yellow', linewidth=2,
             label="predicted_coe2")
plt.plot(targetc[:39,1], H_field[:39,0], marker='*', markerfacecolor='yellow', markersize=2, color='red', linewidth=2,
             label="actual_coe2")
plt.plot(predictc[:15,2], H_field[:15,0], marker='o', markerfacecolor='red', markersize=2, color='grey', linewidth=2,
             label="predicted_coe3")
plt.plot(targetc[:15,2], H_field[:15,0], marker='*', markerfacecolor='yellow', markersize=2, color='orange', linewidth=2,
             label="actual_coe3")
plt.legend()
plt.savefig('coefficient_prediction_waveform.pdf')
plt.show()

#save predicted result and target result to .mat file
import scipy.io as sio
my_dictonary = {}
newkey = 'predict_wave'
my_dictonary[newkey] = predictw
newkey = 'target_wave'
my_dictonary[newkey] = targetw
newkey = 'predict_coefficient'
my_dictonary[newkey] = predictc
newkey = 'target_coefficient'
my_dictonary[newkey] = targetc
sio.savemat('output_laudou_polariton.mat', my_dictonary)




#plot the fft result from the predicted waveform and the target waveform

test = sio.loadmat('dataset_test.mat')
air_ref = test['ref']
sample_input = data[:,0,:175]
sample_predict = np.concatenate((sample_input, predictw[:,0,:]), axis = 1)
sample_target = np.concatenate((sample_input, targetw[:,0,:]), axis = 1)

total_sample_num = len(H_field)

####  Figure Settings ####
font_size = 10
fig = plt.figure()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
offset_step = 0.02
fft_num=2048
spec_list1 = np.zeros([fft_num//2, total_sample_num + 1])
spec_list2 = np.zeros([fft_num//2, total_sample_num + 1])
spec_list3 = np.zeros([fft_num//2, total_sample_num + 1])

trans_offset = np.linspace(start=0, stop=offset_step*total_sample_num, num=total_sample_num,endpoint=True)


for sample_index in range(total_sample_num):
    offset = offset_step*sample_index

    time_predict = sample_predict[sample_index,:]
    time_target = sample_target[sample_index,:]
    time_input = sample_input[sample_index,:]
    time_ref = air_ref[0,1:]

    L = len(time_ref)
    Ts = 0.2e-12
    t_sweep = np.linspace(0, Ts*L, num=L, endpoint=True)
    freq = fftfreq(fft_num, Ts)[:fft_num//2]

    freq_predict = fft(time_predict, n=fft_num)
    freq_target = fft(time_target, n=fft_num)
    freq_input = fft(time_input, n=fft_num)
    freq_air = fft(time_ref, n=fft_num)

    spec1 = np.abs(freq_predict/freq_air)**2 + offset
    spec2 = np.abs(freq_target/freq_air)**2 + offset
    spec3 = np.abs(freq_input/freq_air)**2 + offset

    spec_list1[:,0] = freq
    spec_list1[:,sample_index+1] =  spec1[:fft_num//2] - offset
    spec_list2[:,0] = freq
    spec_list2[:,sample_index+1] =  spec2[:fft_num//2] - offset
    spec_list3[:,0] = freq
    spec_list3[:,sample_index+1] =  spec3[:fft_num//2] - offset

    ax1.fill_between(freq/1e12, offset, spec1[:fft_num//2])
    ax1.set_xlim([0.266, 0.569])
    ax1.set_aspect(aspect=0.5)
    ax1.set_ylim([0,offset_step*total_sample_num])

    ax2.fill_between(freq/1e12, offset, spec2[:fft_num//2])
    ax2.set_xlim([0.266, 0.569])
    ax2.set_aspect(aspect=0.5)
    ax2.set_ylim([0,offset_step*total_sample_num])


    ax3.fill_between(freq/1e12, offset, spec3[:fft_num//2])
    ax3.set_xlim([0.266, 0.569])
    ax3.set_aspect(aspect=0.5)
    ax3.set_ylim([0,offset_step*total_sample_num])


ax1.set_xlabel('Frequency (THz)', fontsize=font_size)
ax1.set_ylabel('Transmission', fontsize=font_size)
ax2.set_xlabel('Frequency (THz)', fontsize=font_size)
ax2.set_ylabel('Transmission', fontsize=font_size)
ax3.set_xlabel('Frequency (THz)', fontsize=font_size)
ax3.set_ylabel('Transmission', fontsize=font_size)
ax1.title.set_text('Predicted result')
ax2.title.set_text('Target result')
ax3.title.set_text('Input truncated result')
plt.savefig('waveform_fft_result.pdf')
plt.show()

print('demo finshed')
