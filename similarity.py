import numpy as np
import Levenshtein
import torch
import pandas as pd
import itertools
from GCNdata import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_bandwidth(interaction_profile):
    squared_sum = np.sum(interaction_profile ** 2)
    n = len(interaction_profile)
    if n * squared_sum == 0:
        return np.inf
    bandwidth = 1 / (n * squared_sum)
    return bandwidth


def gaussian_kernel_similarity(IP1, IP2, bandwidth):
    diff = IP1 - IP2
    norm_squared = np.dot(diff, diff)
    if bandwidth == np.inf:
        return 0.0
    similarity = np.exp(-bandwidth * norm_squared)
    return similarity


def build_gaussian_similarity_matrix(IP, is_lncRNA=True):
    n = IP.shape[0] if is_lncRNA else IP.shape[1]
    GSM = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if is_lncRNA:
                IP_i = IP[i, :]
                IP_j = IP[j, :]
            else:
                IP_i = IP[:, i]
                IP_j = IP[:, j]
            bandwidth = calculate_bandwidth(IP_i) if is_lncRNA else calculate_bandwidth(IP_j)
            GSM[i, j] = gaussian_kernel_similarity(IP_i, IP_j, bandwidth)
    return GSM


def calculate_similarity(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)
    lev_distance = Levenshtein.distance(seq1, seq2)
    similarity = (max(len1, len2) - lev_distance) / max(len1, len2)
    return similarity


def build_similarity_matrix(sequences):
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = calculate_similarity(sequences[i], sequences[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
    return similarity_matrix


def read_expression_profiles(file_path):
    df = pd.read_csv(file_path, sep=' ')
    return df


def express_similarity(profile_a, profile_b):
    pa = np.array(profile_a, dtype=float)
    pb = np.array(profile_b, dtype=float)
    pa_mean = np.mean(pa)
    pb_mean = np.mean(pb)
    pa_diff = pa - pa_mean
    pb_diff = pb - pb_mean
    numerator = np.sum(pa_diff * pb_diff)
    denominator = np.sqrt(np.sum(pa_diff ** 2) * np.sum(pb_diff ** 2))
    correlation = numerator / denominator if denominator != 0 else 0
    similarity = (correlation + 1) / 2
    return similarity

