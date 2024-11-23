import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam

from dataloaders import load_data, MF_loader
from models import MF
from trainers import MF_trainer, MF_test
from modules import arg_parsing


train_log = []

# args 선언
print('arg parsing')
args = arg_parsing()

print('device:',args.device)

# 파일 임포트
print('FILE LOADing...')
data = load_data(args)


# DataLoader
print('DATA LOADing...')
data = MF_loader(args, data)

n_users, n_items = len(data['user2id']), len(data['item2id'])
model = MF(n_users, n_items, args)
model.to(args.device)
loss_fn = nn.BCELoss().to(args.device)
optimizer = Adam(model.parameters(), lr=args.lr)


# trainer
print('TRAINing...')
trainer = MF_trainer
trainer(args, data, model, loss_fn, optimizer)


# test
print('TESTing...')
test = MF_test
test(args, data, model)


# model file 저장

# 학습로그 기록