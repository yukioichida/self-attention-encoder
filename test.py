import torch.nn.functional as F
import torch

fn = F.cross_entropy

label = torch.tensor([2])

predicted = torch.tensor([[0.0, 0.0, 5.]])

print("Loss = {:.4f}".format(fn(predicted, label)))

print("torch.max {}".format(torch.max(predicted, 1)))

# POSITIONAL EMBEDDING JUST LIKE OPEN AI FINE TUNED TRANSFORMER

import numpy as np

# word indexes
batch = 3
max_seq_length = 3
xmb = torch.zeros((batch, 2, max_seq_length, 2), dtype=torch.int32)
xmb[:, :, :max_seq_length, 0] = torch.tensor([11, 23, 123])
xmb[:, :, :max_seq_length, 1] = torch.tensor([1, 2, 3])
print(xmb)
print(xmb.size())
print("reshape")
xmb_reshape = xmb.view(-1, xmb.size(-2), xmb.size(-1))
print(xmb_reshape)
print(xmb_reshape.size())
print("Sum - simulating the positional encoding")
print(xmb_reshape.sum(dim=0))  # sum element for each d in x[d, :, :, :]
print(xmb_reshape.sum(dim=2))  # sum element for each d, each e, each f in x[d, e, f, :]
