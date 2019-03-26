import torch
import torch.nn as nn

class ModelAttention(nn.Module):
    ''' Attention Model '''

    def __init__(self, dim_model)


class MultiHeadAttention(nn.Module):
    ''' Multihead attention '''

    def __init__(self, qtd_head, dim_model, dim_k, dim_v):
        super(MultiHeadAttention, self).__init__()

        self.qtd_head = qtd_head
        self.dim_k = dim_k
        self.dim_k = dim_v

        self.weight_q = nn.Parameter(torch.FloatTensor(n_head, dim_model, dim_k))
        self.weight_k = nn.Parameter(torch.FloatTensor(n_head, dim_model, dim_k))
        self.weight_v = nn.Parameter(torch.FloatTensor(n_head, dim_model, dim_v))
        

