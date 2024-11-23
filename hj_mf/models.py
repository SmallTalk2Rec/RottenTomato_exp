import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset



class MF(nn.Module):
    def __init__(self, n_users:int, n_items:int, args):
        super().__init__()
        self.user_embed = nn.Embedding(n_users+1, args.embed_dims)
        self.item_embed = nn.Embedding(n_items+1, args.embed_dims)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.user_embed.weight.data)
        nn.init.xavier_uniform_(self.item_embed.weight.data)


    def forward(self, user, item):
        """
        uTv: dot product 연산
        """

        xu = self.user_embed(user).reshape(-1,1,64)
        xi = self.item_embed(item).reshape(-1,64,1)

        if not xu.is_contiguous():
            xu = xu.contiguous()
        if not xi.is_contiguous():
            xi = xi.continguous()

        result = torch.bmm(xu,xi)
        result = self.sigmoid(result)

        return result.reshape(-1)
