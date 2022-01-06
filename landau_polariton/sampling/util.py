import os
import torch
import torch.onnx
import matplotlib.pyplot as plt
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cwd = os.getcwd()


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



def evaluate_05(modelw,modelw1,modelw2,test_loader, criterion):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :250]
            targetw = ori_data[:, :, 250:]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[:,:,:], targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data,w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data, w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-ori_data.shape[2] // 2:,:,:], targetw[-ori_data.shape[2] // 2:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss



def evaluate_2(modelw,modelw1,modelw2, test_loader, criterion):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :75]
            targetw = ori_data[:, :, 75:500]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[25:,:,:], targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data,w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data, w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-250:,:,:], targetw[-250:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss


def evaluate_3(modelw,modelw1,modelw2, test_loader, criterion):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :45]
            targetw = ori_data[:, :, 45:330]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[5:,:,:], targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data[5:,:,:],w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data, w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-165:,:,:], targetw[-165:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss


def evaluate_4(modelw,modelw1,modelw2, test_loader, criterion):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :35]
            targetw = ori_data[:, :, 35:250]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[5:,:,:], targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data[5:,:,:],w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data, w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-125:,:,:], targetw[-125:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss


def evaluate_5(modelw,modelw1,modelw2, test_loader, criterion):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :25]
            targetw = ori_data[:, :, 25:200]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[:,:,:], targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data,w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data[:,:,:], w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-100:,:,:], targetw[-100:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss


def evaluate_10(modelw,modelw1,modelw2, test_loader, criterion):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :15]
            targetw = ori_data[:, :, 15:100]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[5:15,:,:], targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data,w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data[:,:,:], w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-50:,:,:], targetw[-50:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss


def evaluate_20(modelw,modelw1,modelw2, test_loader, criterion):
    # Turn on evaluation mode which disables dropout.
    modelw.eval()
    modelw1.eval()
    modelw2.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, :10]
            targetw = ori_data[:, :, 10:50]
            data= data.to(device)
            targetw = targetw.to(device)


            data = data.permute(2, 0, 1)
            targetw = targetw.permute(2, 0, 1)
            batch_size = data.shape[1]

            # waveform process
            w_out1,_ = modelw2(data[5:10,:,:], targetw, batch_size)
            w_out2, _ = modelw1(torch.cat((data[5:10,:,:],w_out1), dim = 0), targetw, batch_size)
            data1 = torch.cat((data[:,:,:], w_out1,w_out2), dim = 0)
            w_out3, _ = modelw(data1, targetw, batch_size)


            output = torch.cat((w_out1, w_out2, w_out3), dim = 0)
            loss = criterion(output[-25:,:,:], targetw[-25:,:,:])
            total_loss += loss.item()

    return ori_data, targetw, output, total_loss