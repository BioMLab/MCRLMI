import numpy as np
import Levenshtein
import torch
import pandas as pd
import itertools
from data_process import *
from similarity import *
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graph(network, p):
    rows, cols = network.shape
    np.fill_diagonal(network, 0)
    PNN = np.zeros((rows, cols))
    graph = np.zeros((rows, cols))

    for i in range(rows):
        idx = np.argsort(-network[i, :])
        PNN[i, idx[:p]] = network[i, idx[:p]]
    for i in range(rows):
        idx_i = np.nonzero(PNN[i, :])[0]
        for j in range(rows):
            idx_j = np.nonzero(PNN[j, :])[0]
            if j in idx_i and i in idx_j:
                graph[i, j] = 1
            elif j not in idx_i and i not in idx_j:
                graph[i, j] = 0
            else:
                graph[i, j] = 0.5
    return graph


def adj_to_edge_index_weighted(adj_matrix):
    edge_indices = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
    edge_weights = adj_matrix[edge_indices[0], edge_indices[1]]
    return edge_indices, edge_weights


def load_adj_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            row = [float(value) for value in line.strip().split(' ')]
            matrix.append(row)
        tensor_matrix = torch.tensor(matrix, dtype=torch.float)
    return tensor_matrix


def process_GCNdata(files):
    adj_matrices = [load_adj_matrix_from_file(file) for file in files]
    num_nodes = adj_matrices[0].size(0)
    num_features = adj_matrices[0].size(1)
    num_matrices = len(adj_matrices)
    edge_indices = []
    edge_weights = []
    for adj in adj_matrices:
        edge_index, edge_weight = adj_to_edge_index_weighted(adj)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
    feature_matrices = [torch.randn(adj_matrix.size(0), adj_matrix.size(1)) for adj_matrix in adj_matrices]
    return adj_matrices, num_nodes, num_features, num_matrices, feature_matrices, edge_indices, edge_weights


def getUseDate(filepath):
    mi_files = ['./database/processedData/data/miRNA_sequences_P.txt',
                './database/processedData/data/miRNA_GIP_P.txt',
                './database/processedData/data/miRNA_express_P.txt']
    mi_adj_matrices, mi_num_nodes, mi_num_features, mi_num_matrices, mi_feature_matrices,\
        mi_edge_indices, mi_edge_weights = process_GCNdata(mi_files)

    lnc_files = ['./database/processedData/data/lncRNA_sequences_P.txt',
                 './database/processedData/data/lncRNA_GIP_P.txt',
                 './database/processedData/data/lncRNA_express_P.txt']
    lnc_adj_matrices, lnc_num_nodes, lnc_num_features, lnc_num_matrices, lnc_feature_matrices,\
        lnc_edge_indices, lnc_edge_weights = process_GCNdata(lnc_files)

    l_m_adj_matrices = read_txt("database/processedData/data/lncRNA-miRNA interaction.txt")
    l_m_adj_matrices = torch.tensor(l_m_adj_matrices, dtype=torch.float32)

    data = {'mi_x': mi_feature_matrices,
            'mi_edge_index': mi_edge_indices,
            'mi_edge_weights': mi_edge_weights,
            'mi_num_nodes': mi_num_nodes,
            'mi_num_features': mi_num_features,
            'mi_num_matrices': mi_num_matrices,
            'lnc_x': lnc_feature_matrices,
            'lnc_edge_index': lnc_edge_indices,
            'lnc_edge_weights': lnc_edge_weights,
            'lnc_num_nodes': lnc_num_nodes,
            'lnc_num_features': lnc_num_features,
            'lnc_num_matrices': lnc_num_matrices,
            'l_m_adj_matrices': l_m_adj_matrices
            }
    return data