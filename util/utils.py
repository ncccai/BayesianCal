import os
import torch
import json

def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    path_prefix = os.path.join('data', 'cmapss', dataset)
    train_data_dir = os.path.join(path_prefix, 'train')
    test_data_dir = os.path.join(path_prefix, 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data

def convert_data(X, y, y_cls, dataset=''):
    if not isinstance(X, torch.Tensor):
        if 'celeb' in dataset.lower():
            X=torch.Tensor(X).type(torch.float32).permute(0, 3, 1, 2)
            y=torch.Tensor(y).type(torch.int64)

        else:
            X=torch.Tensor(X).type(torch.float32)
            y=torch.Tensor(y).type(torch.int64)
            y_cls = torch.Tensor(y_cls).type(torch.int64)

    return X, y, y_cls

def read_user_data(index, data, dataset=''):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, y_train_cls = convert_data(train_data['x'], train_data['y'], train_data['y_cls'], dataset=dataset)
    train_data = [(x, y, y_cls) for x, y, y_cls in zip(X_train, y_train, y_train_cls)]
    X_test, y_test, y_test_cls = convert_data(test_data['x'], test_data['y'], test_data['y_cls'], dataset=dataset)
    test_data = [(x, y, y_cls) for x, y, y_cls in zip(X_test, y_test, y_test_cls)]

    return id, train_data, test_data