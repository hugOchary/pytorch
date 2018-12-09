import torch
from fashionMNIST import hot_ones

test_labels = torch.tensor([9,2,5,4,3,1,6,7,1,2,5], dtype=torch.int32)

print(hot_ones(test_labels, 10))