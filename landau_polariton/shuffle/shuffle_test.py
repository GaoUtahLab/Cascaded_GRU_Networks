# coding: utf-8
import os
import torch
import torch.onnx
from torchvision import datasets, transforms
from torch import nn, optim
from dataset_full_waveform import EMDataset
import scipy.io as sio



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
        self.rnn = nn.GRU(
            input_size=1,
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






def evaluate(model):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :,  100:550]
            targets = ori_data[:, :, 550:1000]
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.permute(2, 0, 1)
            data = data.permute(2, 0, 1)
            batch_size = targets.shape[1]

            output = model(data, targets, batch_size)
            # output = output[400:600, :, :]
            loss = criterion(output, targets)
            total_loss += loss.item()
            if batch_idx % 5 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * data.shape[1], len(test_loader.dataset),
                           100. * batch_idx / len(test_loader),
                    total_loss))
    return total_loss


def train(epoch,model):

    model.train()
    total_loss = 0
    for batch_idx, (ori_data) in enumerate(train_loader):
        optimizer.zero_grad()
        data = ori_data[:, :, 100:550]
        targets = ori_data[:,:,550:1000]
        data = data.to(device)
        targets = targets.to(device)
        targets = targets.permute(2, 0, 1)
        data = data.permute(2, 0, 1)
        batch_size = targets.shape[1]

        output = model(data,targets,batch_size)



        loss = criterion(output, targets)
        # #############################

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * data.shape[1], len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       total_loss))
            total_loss = 0

    return total_loss




filename = 'data_shuffle.mat'
dir = os.path.join(cwd, filename)
criterion = nn.MSELoss()


eval_loss = []
for idx in range(38):
    if device =='cuda':
        model = torch.load('GRU_model_exp_pretrain_500_950.pkl')
    else:
        model = torch.load('GRU_model_exp_pretrain_500_950.pkl', map_location=torch.device('cpu'))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01)

    name_base = 'GRU_model_same_seq_500_950_random_predict'
    name = name_base + '_' + str(idx+1) + '.pkl'
    train_name_base = 'train'
    test_name_base = 'test'
    train_name = train_name_base + str(idx+1)
    test_name = test_name_base + str(idx+1)


    HV_input1 = train_name
    HV_input2 = test_name


    trainset = EMDataset(dir, HV_input1, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    testset = EMDataset(dir, HV_input2, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=11, shuffle=True)

    for epoch in range(1, 150):
        train(epoch,model)
        val_loss = evaluate(model)

        torch.save(model, name)
    eval_loss.append(val_loss)
    my_dictonary = {}
    newkey = 'shuffle_data_eval_loss'
    my_dictonary[newkey] = eval_loss
    sio.savemat('shuffle_data_eval_loss.mat', my_dictonary)
