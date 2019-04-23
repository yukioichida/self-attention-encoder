import numpy as np
import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dim_model):
        # root square of dimension size
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(dim_model, 0.5)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v):
        ''' Returns the softmax scores and attention tensor '''
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.temper
        attention = self.softmax(attention)
        output = torch.bmm(attention, v)
        return output, attention


class MultiHeadAttention(nn.Module):
    ''' Multihead attention '''

    def __init__(self, qty_head, dim_model, dim_k, dim_v):
        super(MultiHeadAttention, self).__init__()

        self.qty_head = qty_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_model = dim_model

        self.weight_q = nn.Parameter(torch.FloatTensor(qty_head, dim_model, dim_k))
        self.weight_k = nn.Parameter(torch.FloatTensor(qty_head, dim_model, dim_k))
        self.weight_v = nn.Parameter(torch.FloatTensor(qty_head, dim_model, dim_v))

        self.attention_model = ScaledDotProductAttention(dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)
        # V vectors of each head are concatenated
        self.projection = nn.Linear(qty_head * dim_v, dim_model)

        torch.nn.init.xavier_normal_(self.weight_q)
        torch.nn.init.xavier_normal_(self.weight_k)
        torch.nn.init.xavier_normal_(self.weight_v)

    def forward(self, q, k, v):
        residual = q

        batch_size, q_len, dim_model = q.size()
        _, k_len, _ = k.size()
        _, v_len, _ = v.size()

        # Reshaping considering number of heads
        q_vector = q.repeat(self.qty_head, 1, 1).view(self.qty_head, -1, self.dim_model)
        k_vector = k.repeat(self.qty_head, 1, 1).view(self.qty_head, -1, self.dim_model)
        v_vector = v.repeat(self.qty_head, 1, 1).view(self.qty_head, -1, self.dim_model)

        q_vector = torch.bmm(q_vector, self.weight_q).view(-1, q_len, self.dim_k)
        k_vector = torch.bmm(k_vector, self.weight_k).view(-1, k_len, self.dim_k)
        v_vector = torch.bmm(v_vector, self.weight_v).view(-1, v_len, self.dim_v)

        outputs, attentions = self.attention_model(q_vector, k_vector, v_vector)

        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)
        outputs = self.projection(outputs)

        return self.layer_norm(outputs + residual), attentions


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, dim_hidden, dim_inner_hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_1 = nn.Linear(dim_hidden, dim_inner_hidden)  # position-wise
        self.layer_2 = nn.Linear(dim_inner_hidden, dim_hidden)  # position-wise
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.layer_1(x))
        output = self.layer_2(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    ''' Transformer encoder layer '''

    def __init__(self, dim_model, dim_inner_hidden, qty_head, dim_k, dim_v):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(qty_head, dim_model, dim_k, dim_v)
        self.feedforward = PositionwiseFeedForward(dim_model, dim_inner_hidden)

    def forward(self, input_tensor):
        output, attention = self.self_attention(input_tensor, input_tensor, input_tensor)
        output = self.feedforward(output)
        return output, attention


def position_encoding_init(positions, dim_word_vector):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_encoder = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim_word_vector) for j in range(dim_word_vector)]
        if pos != 0 else np.zeros(dim_word_vector) for pos in range(positions)])

    position_encoder[1:, 0::2] = np.sin(position_encoder[1:, 0::2])  # dim 2i
    position_encoder[1:, 1::2] = np.cos(position_encoder[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_encoder).type(torch.FloatTensor)


class TransformerEncoder(nn.Module):
    ''' A neural network Transformer Encoder '''

    def __init__(self, vocab_size, max_sequence_length, qty_encoder_layer=1, qty_attention_head=2,
                 dim_k=32, dim_v=32, dim_word_vector=128, dim_model=128, dim_inner_hidden=256, output_size=3):
        super(TransformerEncoder, self).__init__()
        positions = max_sequence_length + 1  # counting UNK

        self.max_sequence_length = max_sequence_length
        self.dim_model = dim_model

        # Embedding containing sentence order information
        self.position_encoder = nn.Embedding(positions, dim_word_vector, padding_idx=0)
        self.position_encoder.weight.data = position_encoding_init(positions, dim_word_vector)

        # Embedding vector of words. TODO: test with word2vec
        self.word_embedding_layer = nn.Embedding(vocab_size, dim_word_vector, padding_idx=0)

        # Create a set of encoder layers, given the quantity informed in
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim_model, dim_inner_hidden, qty_attention_head, dim_k, dim_v)
            for _ in range(qty_encoder_layer)
        ])

        # Output layer, which convert the encoded tensors into the label representation.
        self.output_layer = nn.Linear(max_sequence_length * dim_model, output_size)

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        position_parameters = set(map(id, self.position_encoder.parameters()))
        return (p for p in self.parameters() if id(p) not in position_parameters)

    def forward(self, sequence):
        positions = self.get_positions(sequence)
        # lookup word embedding layer
        word_embedding = self.word_embedding_layer(sequence)
        # lookup positional encoding
        positional_encoding = self.position_encoder(positions)

        encoder_output = word_embedding + positional_encoding

        for encoder_layer in self.encoder_layers:
            encoder_output, attentions = encoder_layer(encoder_output)

        batch_size, q_len, dim_model = encoder_output.size()

        # batch size x number of classes
        output = self.output_layer(encoder_output.view((batch_size, -1)))
        # [probability of being the first class, probability of being the second class] thas why is two dimensional
        return output

    def get_positions(self, sequence):
        PADDING = 0
        positions = [[pos + 1 if word != PADDING else 0 for pos, word in enumerate(instance)] for instance in sequence]
        return torch.autograd.Variable(torch.LongTensor(positions), volatile=False)
