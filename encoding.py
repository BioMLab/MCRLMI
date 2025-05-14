import numpy as np
import Levenshtein
import torch
import pandas as pd
import itertools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maxLength(file_path):
    miRNA_sequences = []
    lncRNA_sequences = []
    labels = []
    removed_space_count = 0
    with open(file_path, 'r') as f:
        lines = [line.strip().split(',') for line in f]
    for line in lines:
        if len(line) != 5:
            continue
        miRNA_name, lncRNA_name, miRNA_seq, lncRNA_seq, label_raw = line
        miRNA_sequences.append(miRNA_seq.replace('U', 'T'))
        lncRNA_sequences.append(lncRNA_seq.replace('U', 'T'))
        label_raw = label_raw.strip()
        try:
            label = int(label_raw)
        except ValueError:
            continue
        if label_raw != line[4]:
            removed_space_count += 1
        labels.append(label)
    mi_max_length = max(len(seq) for seq in miRNA_sequences) if miRNA_sequences else 0
    lnc_max_length = max(len(seq) for seq in lncRNA_sequences) if lncRNA_sequences else 0
    return mi_max_length, lnc_max_length


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for seq in sequences:
        n = max_length // len(seq)
        c = max_length % len(seq)
        padded_seq = (seq * n) + seq[:c]
        padded_sequences.append(padded_seq)
    return padded_sequences, max_length


def encode_high_order_one_hot(sequences, k):
    bases = ['A', 'G', 'C', 'T']
    k_mers = []
    for j in range(1, k + 1):
        k_mers.extend(''.join(p) for p in itertools.product(bases, repeat=j))
    encoding_map_length = len(k_mers)
    k_mer_to_index = {k_mer: i for i, k_mer in enumerate(k_mers)}
    k_mer_to_vector = {
        k_mer: np.eye(1, encoding_map_length, i, dtype=int)[0]
        for i, k_mer in enumerate(k_mers)
    }
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [
            k_mer_to_vector[seq[i:i + k]]
            for i in range(0, len(seq) - k + 1, k)
        ]
        remaining = len(seq) % k
        if remaining != 0:
            last_k_mer = seq[-remaining:]
            if last_k_mer in k_mer_to_vector:
                encoded_seq.append(k_mer_to_vector[last_k_mer])
            else:
                encoded_seq.append(np.zeros(encoding_map_length))
        encoded_sequences.append(np.array(encoded_seq))
    return encoded_sequences, k_mer_to_index, k_mer_to_vector


def process_file(file_path, k, mi_max_length, lnc_max_length):
    removed_space_count = 0
    with open(file_path, 'r') as f:
        lines = [line.strip().split(',') for line in f]
    miRNA_sequences = []
    lncRNA_sequences = []
    labels = []
    for line in lines:
        if len(line) != 5:
            continue
        miRNA_name, lncRNA_name, miRNA_seq, lncRNA_seq, label_raw = line
        miRNA_sequences.append(miRNA_seq.replace('U', 'T'))
        lncRNA_sequences.append(lncRNA_seq.replace('U', 'T'))
        label_raw = label_raw.strip()
        if label_raw != line[4]:
            removed_space_count += 1
        try:
            label = int(label_raw)
        except ValueError:
            continue
        labels.append(label)

    padded_miRNA, max_length_miRNA = pad_sequences(miRNA_sequences, mi_max_length)
    padded_lncRNA, max_length_lncRNA = pad_sequences(lncRNA_sequences, lnc_max_length)
    encoded_miRNA, miRNA_k_mer_to_index, miRNA_k_mer_to_vector = encode_high_order_one_hot(padded_miRNA, k)
    encoded_lncRNA, lncRNA_k_mer_to_index, lncRNA_k_mer_to_vector = encode_high_order_one_hot(padded_lncRNA, k)

    encoded_miRNA = np.array(encoded_miRNA)
    encoded_miRNA = torch.from_numpy(encoded_miRNA).float().to(device)
    encoded_lncRNA = np.array(encoded_lncRNA)
    encoded_lncRNA = torch.from_numpy(encoded_lncRNA).float().to(device)

    label_integers = [int(label) for label in labels]
    y_labels = torch.tensor(label_integers, dtype=torch.float).to(device)

    mi_seq_len = encoded_miRNA.shape[1]
    lnc_seq_len = encoded_lncRNA.shape[1]
    d_input = encoded_miRNA.shape[2]

    return encoded_miRNA, mi_seq_len, encoded_lncRNA, lnc_seq_len, y_labels, d_input

