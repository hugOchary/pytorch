import torch
import torch.nn as nn
import numpy as np
from CNN_classes import Network

test_tensor = torch.ones([10,1,28,28])

network = Network()

print(network(test_tensor))
print(network.conv_layer1.is_cuda)