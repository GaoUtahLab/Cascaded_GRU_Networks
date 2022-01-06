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
    plt.plot(predictw[0, 0, :], marker='o', markerfacecolor='red', markersize=1, color='blue', linewidth=1,
             label="predicted")
    plt.plot(targetw[0, 0, :], marker='*', markerfacecolor='yellow', markersize=1, color='purple', linewidth=1,
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

def evaluate_25(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model00.eval()
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 100:125]
            targets = ori_data[:, :, 125:1700]
            data = data.permute(2, 0, 1)
            targets = targets.permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[1]
            output000 = model(data, targets, batch_size)
            data00 = torch.cat((data, output000), dim=0)
            output00 = model00(data00, targets, batch_size)
            data0 = torch.cat((data00, output00), dim=0)
            output0 = model0(data0, targets, batch_size)  #100:200 -> 200: 300
            data1 = torch.cat((data0,output0), dim=0)
            output1 = model1(data1, targets, batch_size) #100:300 -> 300:500
            output2 = model2(output1, targets, batch_size) #300:500 -> 500:700
            output3 = model3(output2,targets,batch_size) #500:700 -> 700:900
            data2 = torch.cat((output2, output3), dim=0) #500:900
            output4 = model4(data2,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output000, output00, output0, output1, output2, output3, output4, output5), dim=0)
            # total_loss += criterion(output, targets)
            total_loss += criterion(output[-400:, :, :], targets[-400:, :, :])
    return data, output, targets, ori_data, total_loss.item()


def evaluate_50(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader):
    # Turn on evaluation mode which disables dropout.
    model00.eval()
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 100:150]
            targets = ori_data[:, :, 150:1700]
            data = data.permute(2, 0, 1)
            targets = targets.permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[1]
            output00 = model00(data, targets, batch_size)
            data0 = torch.cat((data, output00), dim=0)
            output0 = model0(data0, targets, batch_size)  #100:200 -> 200: 300
            data1 = torch.cat((data0,output0), dim=0)
            output1 = model1(data1, targets, batch_size) #100:300 -> 300:500
            output2 = model2(output1, targets, batch_size) #300:500 -> 500:700
            output3 = model3(output2,targets,batch_size) #500:700 -> 700:900
            data2 = torch.cat((output2, output3), dim=0) #500:900
            output4 = model4(data2,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output00, output0, output1, output2, output3, output4, output5), dim=0)
            # total_loss += criterion(output, targets)
            total_loss += criterion(output[-400:, :, :], targets[-400:, :, :])
    return data, output, targets, ori_data, total_loss.item()


def evaluate_100(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader):
    # Turn on evaluation mode which disables dropout.
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 100:200]
            targets = ori_data[:, :, 200:1700]
            data = data.permute(2, 0, 1)
            targets = targets.permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[1]
            output0 = model0(data, targets, batch_size)  #100:200 -> 200: 300
            data1 = torch.cat((data,output0), dim=0)
            output1 = model1(data1, targets, batch_size) #100:300 -> 300:500
            output2 = model2(output1, targets, batch_size) #300:500 -> 500:700
            output3 = model3(output2,targets,batch_size) #500:700 -> 700:900
            data2 = torch.cat((output2, output3), dim=0) #500:900
            output4 = model4(data2,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output0, output1, output2, output3, output4, output5), dim=0)
            # total_loss += criterion(output, targets)
            total_loss += criterion(output[-400:,:,:], targets[-400:,:,:])
    return data, output, targets, ori_data, total_loss.item()


def evaluate_200(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader):
    # Turn on evaluation mode which disables dropout.
    model0.eval()
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
            # output0 = model0(data, targets, batch_size)  #100:200 -> 200: 300
            # data1 = torch.cat((data,output0), dim=0)
            output1 = model1(data, targets, batch_size) #100:300 -> 300:500
            output2 = model2(output1, targets, batch_size) #300:500 -> 500:700
            output3 = model3(output2,targets,batch_size) #500:700 -> 700:900
            data2 = torch.cat((output2, output3), dim=0) #500:900
            output4 = model4(data2,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output1, output2, output3, output4, output5), dim=0)
            total_loss += criterion(output[-400:,:,:], targets[-400:,:,:])


    return data, output, targets, ori_data, total_loss.item()



def evaluate_400(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader):
    # Turn on evaluation mode which disables dropout.
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 300:500]
            targets = ori_data[:, :, 500:1700]
            data = data.permute(2, 0, 1)
            targets = targets.permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[1]
            # output0 = model0(data, targets, batch_size)  #100:200 -> 200: 300
            # data1 = torch.cat((data,output0), dim=0)
            # output1 = model1(data, targets, batch_size) #100:300 -> 300:500
            output2 = model2(data, targets, batch_size) #300:500 -> 500:700
            output3 = model3(output2,targets,batch_size) #500:700 -> 700:900
            data2 = torch.cat((output2, output3), dim=0) #500:900
            output4 = model4(data2,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output2, output3, output4, output5), dim=0)
            # total_loss += criterion(output, targets)
            total_loss += criterion(output[-400:,:,:], targets[-400:,:,:])
    return data, output, targets, ori_data, total_loss.item()


def evaluate_600(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader):
    # Turn on evaluation mode which disables dropout.
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 500:700]
            targets = ori_data[:, :, 700:1700]
            data = data.permute(2, 0, 1)
            targets = targets.permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[1]
            # output0 = model0(data, targets, batch_size)  #100:200 -> 200: 300
            # data1 = torch.cat((data,output0), dim=0)
            # output1 = model1(data, targets, batch_size) #100:300 -> 300:500
            # output2 = model2(data, targets, batch_size) #300:500 -> 500:700
            output3 = model3(data,targets,batch_size) #500:700 -> 700:900
            data2 = torch.cat((data, output3), dim=0) #500:900
            output4 = model4(data2,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output3, output4, output5), dim=0)
            # total_loss += criterion(output, targets)
            total_loss += criterion(output[-400:,:,:], targets[-400:,:,:])
    return data, output, targets, ori_data, total_loss.item()


def evaluate_800(model, model00, model0, model1, model2, model3, model4, model5, criterion, test_loader):
    # Turn on evaluation mode which disables dropout.
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (ori_data) in enumerate(test_loader):
            data = ori_data[:, :, 500:900]
            targets = ori_data[:, :, 900:1700]
            data = data.permute(2, 0, 1)
            targets = targets.permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[1]
            # output0 = model0(data, targets, batch_size)  #100:200 -> 200: 300
            # data1 = torch.cat((data,output0), dim=0)
            # output1 = model1(data, targets, batch_size) #100:300 -> 300:500
            # output2 = model2(data, targets, batch_size) #300:500 -> 500:700
            # output3 = model3(output2,targets,batch_size) #500:700 -> 700:900
            # data2 = torch.cat((output2, output3), dim=0) #500:900
            output4 = model4(data,targets,batch_size) #500:900 -> 900:1300
            output5 = model5(output4,targets,batch_size) #900:1300 -> 1300:1700
            output = torch.cat((output4, output5), dim=0)
            # total_loss += criterion(output, targets)
            total_loss += criterion(output[-400:,:,:], targets[-400:,:,:])
    return data, output, targets, ori_data, total_loss.item()
