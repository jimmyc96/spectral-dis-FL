# python version 3.7.1
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np 


def compute_fft_of_weights(weight_tensor):
    # Flatten the weight tensor if it's more than 2D
    if len(weight_tensor.shape) > 2:
        weight_tensor = weight_tensor.view(weight_tensor.shape[0], -1)

    if len(weight_tensor.shape) == 2:
        # For 2D tensors, apply 2D FFT
        fft_weights = np.fft.fft2(weight_tensor.cpu().numpy(), axes=[0, 1])
    elif len(weight_tensor.shape) == 1:
        # For 1D tensors, apply 1D FFT
        fft_weights = np.fft.fft(weight_tensor.cpu().numpy())
    else:
        raise ValueError("Unsupported tensor shape for FFT: {}".format(weight_tensor.shape))

    return fft_weights.flatten()


def FFT_weights_cal(model):
    # List to store all FFT-transformed weights
    all_fft_weights = []

    # Iterate through layers and compute FFT for layers with weights
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight_tensor = layer.weight.data
            fft_weights = compute_fft_of_weights(weight_tensor)
            all_fft_weights.extend(fft_weights)

    # Convert the list to a numpy array
    all_fft_weights_array = np.array(all_fft_weights)
    return all_fft_weights_array




def spectral_cal(model, input=True):
    F_weights=FFT_weights_cal(model)
    prob_dist = np.abs(F_weights)
    prob_dist /= np.sum(prob_dist)

    prob_dist = torch.tensor(prob_dist, dtype=torch.float)

    if input == True:
        prob_dist = F.log_softmax(prob_dist, dim=0)
    return prob_dist



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()  # loss function -- cross entropy
        self.loss_func_kl = nn.KLDivLoss(reduction="batchmean")
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, seed, w_g, epoch, mu=1, lr=None):
        net_glob = w_g

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                
                labels = labels.long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                
                spec_w = spectral_cal(net,input=True)
                spec_l = spectral_cal(net_glob,input=False)
                len_trunc_w = int(len(spec_w) * self.args.ratio)
                len_trunc_l = int(len(spec_l) * self.args.ratio)
                trunc_spec_w, trunc_spec_l = spec_w[0:len_trunc_w], spec_l[0:len_trunc_l]

                if batch_idx > 0:
                    w_diff = torch.tensor(0.).to(self.args.device)
                    l_reg = self.loss_func_kl(trunc_spec_w, trunc_spec_l)
                    loss += self.args.beta * mu * l_reg

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_per(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()  # loss function -- cross entropy
        self.loss_func_kl = nn.KLDivLoss()
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, seed, w_g, epoch, mu=1, lr=None):
        net_glob = w_g

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                
                labels = labels.long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                
                spec_w = spectral_cal(net, input=False)
                spec_l = spectral_cal(net_glob, input=True)
                # trunc_spec_w, trunc_spec_l = spec_w[0:len(spec_w)/2], spec_l[0:len(spec_l)/2]

                if batch_idx > 0:
                    w_diff = torch.tensor(0.).to(self.args.device)
                    l_reg = self.loss_func_kl(spec_l, spec_w)
                    loss += self.args.beta * mu * l_reg
                
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            if args.model == "googlenet":
                outputs = net(images).logits
            else:
                outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


def localtest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            if args.model == "googlenet":
                outputs = net(images).logits
            else:
                outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc