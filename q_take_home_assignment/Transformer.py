import numpy as np
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoder(nn.Module):  # passed successfully
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model).cuda()
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).cuda()
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model).cuda()
        self.v_linear = nn.Linear(d_model, d_model).cuda()
        self.k_linear = nn.Linear(d_model, d_model).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.out = nn.Linear(d_model, d_model).cuda()

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9).cuda()
        scores = F.softmax(scores, dim=-1).cuda()

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v).cuda()
    return output


class FeedForward(nn.Module):  # passed succesfully
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.linear_2 = nn.Linear(d_ff, d_model).cuda()

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x))).cuda()
        x = self.linear_2(x).cuda()
        return x


class Norm(nn.Module):  # passed succesfully
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size,device=device)).cuda()
        self.bias = nn.Parameter(torch.zeros(self.size,device=device)).cuda()
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model).cuda()
        self.norm_2 = Norm(d_model).cuda()
        self.attn = MultiHeadAttention(heads, d_model).cuda()
        self.ff = FeedForward(d_model).cuda()
        self.dropout_1 = nn.Dropout(dropout).cuda()
        self.dropout_2 = nn.Dropout(dropout).cuda()

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, N, heads, d_model):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N) #####changed
        self.norm = Norm(d_model).cuda()

    def forward(self, x, mask):
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return x

class Model(nn.Module):

    def __init__(self, N, heads, d_model):
        super().__init__()
        self.MH1 = Encoder(N, heads, d_model)
        self.MH2 = Encoder(N, heads, d_model)
        self.crossattn1 = MultiHeadAttention(heads, d_model)
        self.crossattn2 = MultiHeadAttention(heads, d_model)
        self.linear1 = nn.Linear(d_model,d_model)
        self.linear2 = nn.Linear(d_model,d_model)
        self.ff1 = FeedForward(d_model)
        self.ff2 = FeedForward(d_model)

    def forward(self, x1, x2):
        out1 = self.MH1(x1, mask)
        out2 = self.MH2(x2, mask)
        out11 = self.crossattn1(out1,out2,out2)
        out22 = self.crossattn2(out2,out1,out1)
        out11 = self.linear1(out11)
        out22 = self.linear2(out22)
        out1 = self.ff1(out11)
        out2 = self.ff2(out22)
        return out1, out2

# d_model = 512
# heads = 8
# N = 6
# inp = torch.randn(3,80,512)
mask = None
# mask = get_pad_mask(inp, 512)
# e = Encoder(N, heads, d_model).cuda()
# e = Model(N, heads, d_model).cuda()
# print(e(inp.cuda(), inp.cuda())[1].shape)