import torch
import torch.nn as nn
import numpy as np
from CNN_classes import Network

test_tensor = torch.ones([10,1,28,28]).cuda()

network = Network().cuda()

print(network(test_tensor))