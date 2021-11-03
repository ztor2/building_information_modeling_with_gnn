import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
import random as rd

def load_data(dataset, names):
    objects = []
    for i in range(len(names)):
        with open("{}.{}".format(dataset, names[i]), 'rb') as f:
            objects.append(pkl.load(f))
    adj = preprocess_graph(objects[0])
    feature = preprocess_feature(objects[1])
    labels = labels_encode(objects[2])
    return adj, feature, labels

def preprocess_feature(feature): # row-sum normalization needed only in case of bag-of-words type data 
#     rowsum = np.array(feature.sum(1)) 
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     feature = r_mat_inv.dot(feature)
    feature = torch.FloatTensor(np.array(feature.todense()))
    return feature

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def labels_encode(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    labels = torch.LongTensor(np.where(labels_onehot)[1])
    return labels

def split(data_len: int, train: int, val: int, test: int):
    idx_train = rd.sample(range(data_len), train); remain= [i for i in range(data_len) if i not in idx_train]
    idx_val = rd.sample(remain, val);              remain= [i for i in range(data_len) if i not in idx_train+idx_val]
    idx_test = rd.sample(remain, test)
    return idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)