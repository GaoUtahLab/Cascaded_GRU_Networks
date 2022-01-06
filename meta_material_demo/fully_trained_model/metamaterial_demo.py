# coding: utf-8
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_test import EMDataset
import matplotlib.pyplot as plt




device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Load data
###############################################################################
cwd = os.getcwd()
filename = 'Ex_data_average_final_set.mat'   ##load the .mat file to the current directory and change the name here
HV_input1 = 'Ex_test_a'                         ##variable name for the HV in the .mat.file
dir = os.path.join(cwd,filename)


eval_batch_size = 250
test = EMDataset(dir, HV_input1, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=eval_batch_size, shuffle=True)

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



def evaluate():
    # Turn on evaluation mode which disables dropout.
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 100:300]
            targets = ori_data[:, :, 300:1700]
            data = data.permute(2, 0, 1)
            targets = targets.permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[1]
            output1 = model1(data, targets, batch_size) #100:300 -> 300:500
            output2 = model2(output1, targets, batch_size) #300:500 -> 500:700
            output3 = model3(output2,targets,batch_size) #500:700 -> 700:900
            data2 = torch.cat((output2, output3), dim=0) #500:900
            output4 = model4(data2,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output1, output2, output3, output4, output5), dim=0)
            total_loss += criterion(output, targets)

    return data, output, targets, ori_data, total_loss / (batch_idx + 1)




if device =='cuda':
    model1 = torch.load('GRU_model_same_seq_200_400_final.pkl')
    model2 = torch.load('GRU_model_same_seq_400_600_final.pkl')
    model3 = torch.load('GRU_model_same_seq_600_800_final.pkl')
    model4 = torch.load('GRU_model_same_seq_800_1200_final.pkl')
    model5 = torch.load('GRU_model_same_seq_1200_1600_final.pkl')

else:
    model1 = torch.load('GRU_model_same_seq_200_400_final.pkl', map_location=torch.device('cpu'))
    model2 = torch.load('GRU_model_same_seq_400_600_final.pkl', map_location=torch.device('cpu'))
    model3 = torch.load('GRU_model_same_seq_600_800_final.pkl', map_location=torch.device('cpu'))
    model4 = torch.load('GRU_model_same_seq_800_1200_final.pkl', map_location=torch.device('cpu'))
    model5 = torch.load('GRU_model_same_seq_1200_1600_final.pkl', map_location=torch.device('cpu'))


##evaluate the model
criterion = nn.MSELoss()

model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)
model5.to(device)
data, predict, target, total_data, loss = evaluate()

data = data.permute(1,2,0)
predict = predict.permute(1,2,0)
target = target.permute(1,2,0)
predict = predict.cpu().detach().numpy()
target = target.cpu().detach().numpy()
data = data.cpu().detach().numpy()


#one sample result plot
plt.plot(predict[0,0,:], marker='o', markerfacecolor='red', markersize=2, color='blue', linewidth=1,
             label="predicted")
plt.plot(target[0,0,:], marker='*', markerfacecolor='yellow', markersize=2, color='purple', linewidth=1,
             label="actual")
plt.legend()
plt.savefig('sample_raw_prediction_waveform.pdf')
plt.close()


#save predicted result and target result to .mat file
import scipy.io as sio
my_dictonary = {}
newkey = 'predict'
my_dictonary[newkey] = predict
newkey = 'target'
my_dictonary[newkey] = target
sio.savemat('test_output_metamaterial.mat', my_dictonary)
print('metamaterial demo finished')
