import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Transformer import Model
import torch.nn.functional as F
from BertEmbedding import BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 768
        heads = 8
        N = 6
        mask = None
        self.model = Model(N, heads, d_model).cuda()

    def forward(self, sent1, sent2):    #sent1 and sent2 are tensors
        seqlen1 = sent1.shape[-2]
        seqlen2 = sent2.shape[-2]
        outputs = self.model(sent1,sent2)
        pooled1 = F.max_pool2d(outputs[0], kernel_size=(seqlen1,1))
        pooled2 = F.max_pool2d(outputs[1], kernel_size=(seqlen2,1))
        # print(pooled.shape)
        return pooled1.squeeze(dim=-2), pooled2.squeeze(dim=-2)

# p = Pooling()
# print(p(2,2)[0].shape)
# print(p(2,2)[1].shape)


class WeightFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # self.out = nn.Linear(2*768, 2)
        self.out = nn.Linear(5*768, 1)

    def forward(self, X1, X2):
        X1 = torch.tanh(X1)
        X2 = torch.tanh(X2)
        res = torch.cat([X1,X2,abs(X1-X2), torch.mul(X1,X2), (X1+X2)/2],dim=-1)
        res = self.out(res)
        return torch.sigmoid(res)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = Pooling()
        self.wf = WeightFusion()
        self.bert = BERT()

    def forward(self, X1, X2):
        # X1 = torch.randn(1, 80, 768).cuda()
        # X2 = torch.randn(1, 80, 768).cuda()

        X1 = self.bert(X1)
        X2 = self.bert(X2)
        out1, out2 = self.pool(X1, X2)
        print(out1.shape,out2.shape)
        res = self.wf(out1,out2).squeeze(0)
        return res


n = Network().cuda()
print(n("I am good", "how are you"))