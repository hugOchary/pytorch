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

num_epochs = 200

torch.set_printoptions(linewidth=120)

network = Network()

criterion = nn.BCELoss()
optimizer = optim.Adam(network.parameters())

def hot_ones(labels, size):
    return torch.as_tensor(
        np.array(
            [[int(index == label.item()) for index in range(size)] for label in labels]
        ), dtype = torch.float32
    )

def train_network(optimizer, datas, labels):
    # set optimizer back to zero
    optimizer.zero_grad()

    #prediction step
    prediction = network(datas)

    #loss computation and backpropagation
    target_labels = hot_ones(labels, 10)
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

    for epoch in range(num_epochs):
        for n_batch, batch in enumerate(train_loader):
            datas, labels = batch
            error = train_network(optimizer, datas, labels)
            if n_batch == 40:
                print(" At epoch ", epoch, "and batch ", n_batch, " we get the error ", error)

    torch.save(network.state_dict(), './models/fashionMNIST')

