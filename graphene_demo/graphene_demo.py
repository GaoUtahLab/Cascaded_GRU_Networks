# coding: utf-8
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_p3_test_final import EMDataset
import matplotlib.pyplot as plt




device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Load data
###############################################################################
cwd = os.getcwd()
filename = 'dataset_app3_new_5_diff.mat'##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)                       ##variable name for the HV in the .mat.file
HV_input1 = 'test_m5'
dir = os.path.join(cwd,filename)




eval_batch_size = 250
test = EMDataset(dir, HV_input1, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test,batch_size=200, shuffle=True)

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
    model0.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, ori_data in enumerate(test_loader):
            data = ori_data[:, :, :150]
            targets = ori_data[:, :, 150:]
            data = data.to(device)
            targets = targets.to(device)
            data = data.permute(2, 0, 1)
            data_input1 = data[:150, :, :]
            batch_size = targets.shape[1]
            targets = targets.permute(2, 0, 1)
            output1 = model0(data_input1,targets,batch_size)
            data2 = torch.cat((data_input1, output1), dim=0)
            output2 = model(data2,targets,batch_size)
            output = torch.cat((output1, output2), dim=0)
            total_loss = criterion(output[-300:,:,:], targets[-300:,:,:])
    return data, output, targets, total_loss.item()




if device =='cuda':
    model0 = torch.load('GRU_model_same_seq_150_normal_new_m5.pkl')
    model = torch.load('GRU_model_same_seq_300_normal_new_m5.pkl')
else:
    model0 = torch.load('GRU_model_same_seq_150_normal_new_m5.pkl',map_location=torch.device('cpu'))
    model = torch.load('GRU_model_same_seq_300_normal_new_m5.pkl',map_location=torch.device('cpu'))



criterion = nn.MSELoss()
model.to(device)
data, predict, target, loss = evaluate()
data = data.permute(1,2,0)
predict = predict.permute(1,2,0)
target = target.permute(1,2,0)
predict = predict.cpu().detach().numpy()
target = target.cpu().detach().numpy()
data = data.cpu().detach().numpy()
print('MSE Loss: {:5.6f}'.format(loss))

#one sample result plot
plt.plot(predict[5,0,:], marker='o', markerfacecolor='red', markersize=2, color='blue', linewidth=2,
             label="predicted")
plt.plot(target[5,0,:], marker='*', markerfacecolor='yellow', markersize=2, color='purple', linewidth=2,
             label="actual")
plt.legend()
plt.savefig('sample_raw_prediction_waveform.pdf')
plt.show()


#save predicted result and target result to .mat file
import scipy.io as sio
my_dictonary = {}
newkey = 'predict'
my_dictonary[newkey] = predict
newkey = 'target'
my_dictonary[newkey] = target
sio.savemat('test_output_graphene.mat', my_dictonary)

print('demo finished')
