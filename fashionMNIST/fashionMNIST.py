import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../')

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb

from CNN_classes import Network

num_epochs = 10

torch.set_printoptions(linewidth=120)

network = Network().cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(network.parameters())

save = False

device = torch.device("cuda:0")
if device:
    print(device)

def hot_ones(labels, size):
    return torch.as_tensor(
        np.array(
            [[int(index == label.item()) for index in range(size)] for label in labels]
        ), dtype = torch.float32
    )

def train_network(optimizer, datas, labels):

    #datas.to(device)
    #labels.to(device)

    # set optimizer back to zero
    optimizer.zero_grad()

    #prediction step
    prediction = network(datas.cuda())

    #loss computation and backpropagation
    target_labels = hot_ones(labels, 10).cuda()
    error = criterion(prediction, target_labels)
    error.backward()

    # weight update
    optimizer.step()

    return error

if __name__ == "__main__":
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_set
        ,batch_size=1000
        ,shuffle=True
    )

    error_mean = 0
    for epoch in range(num_epochs):
        error_mean = 0
        for n_batch, batch in enumerate(train_loader):

            datas, labels = batch
            error = train_network(optimizer, datas, labels)

            error_mean += error.item()

        error_mean = error_mean/60
        print(" At the end of epoch ", epoch," we get the error ", error_mean)
    if save:
        torch.save(network.state_dict(), '../models/fashionMNIST')

