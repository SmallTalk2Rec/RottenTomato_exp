import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch


def load_data(args):

    data = {}
    data['X_train'] = pd.read_csv(args.data_path + 'MF_X_train.csv')
    data['Y_train'] = pd.read_csv(args.data_path + 'MF_Y_train.csv')

    data['X_valid'] = pd.read_csv(args.data_path + 'MF_X_val.csv')
    data['Y_valid'] = pd.read_csv(args.data_path + 'MF_Y_val.csv')

    data['X_test'] = pd.read_csv(args.data_path + 'MF_X_test.csv')
    data['Y_test'] = pd.read_csv(args.data_path + 'MF_Y_test.csv')
    
    return data



def MF_loader(args, data):
    '''
    유저와 아이템의 id만 뱉는 방식의 데이터로더
    [전처리 + dataset + dataloader]

    criticName: uid (userid)
    id: iid (itemid)
    fresh/rotten: 1/0 (binary)

    return: LongTensor([uid number, iid number, 1/0])
    '''
    

    # LabelEncoding -> Out Of Dictionary problem 세게 있네
    user2id = {u:i for i,u in enumerate(data['X_train']['criticName'].unique())}
    item2id = {item:i for i,item in enumerate(data['X_train']['id'].unique())}
    fresh2id = {'fresh':1, 'rotten':0}

    id2user = {i:u for i,u in user2id.items()}
    id2item = {i:item for i,item in item2id.items()}
    id2fresh = {1:'fresh', 0:'rotten'}

    data['X_train']['criticName'] = data['X_train']['criticName'].map(user2id)
    data['X_train']['id'] = data['X_train']['id'].map(item2id)
    data['Y_train'] = data['Y_train']['reviewState'].map(fresh2id)

    data['X_valid']['criticName'] = data['X_valid']['criticName'].map(user2id)
    data['X_valid']['id'] = data['X_valid']['id'].map(item2id)
    data['Y_valid'] = data['Y_valid']['reviewState'].map(fresh2id)
    
    data['X_test']['criticName'] = data['X_test']['criticName'].map(user2id)
    data['X_test']['id'] = data['X_test']['id'].map(item2id)
    data['Y_test'] = data['Y_test']['reviewState'].map(fresh2id)

    
    # LongTensor화
    train_dataset = TensorDataset(torch.LongTensor(data['X_train']['criticName'].values), torch.LongTensor(data['X_train']['id'].values), torch.FloatTensor(data['Y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid']['criticName'].values), torch.LongTensor(data['X_valid']['id'].values), torch.FloatTensor(data['Y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['X_test']['criticName'].values), torch.LongTensor(data['X_test']['id'].values), torch.FloatTensor(data['Y_test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    data['user2id'], data['item2id'], data['id2user'], data['id2item'], data['fresh2id'], data['id2fresh'] = user2id, item2id, id2user, id2item, fresh2id, id2fresh

    return data