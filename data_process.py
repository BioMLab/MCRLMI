import numpy as np
import Levenshtein
import torch
import pandas as pd
import itertools
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split( )
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data


def read_sequences(filepath):
    sequences = []
    with open(filepath, 'r') as file:
        for line in file:
            sequences.append(line.strip())
    return sequences


def move_tensors_to_device(data, device):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        elif isinstance(value, dict):
            data[key] = move_tensors_to_device(value, device)
    return data


def read_fasta_file(fasta_file_path):
    data = []
    removed_space_count = 0
    with open(fasta_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            miRNA_name = parts[0]
            lncRNA_name = parts[1]
            miRNA_sequence = parts[2]
            lncRNA_sequence = parts[3]
            label_raw = parts[4].strip()
            if label_raw != parts[4]:
                removed_space_count += 1
            try:
                label = int(label_raw)
            except ValueError:
                continue
            data.append([miRNA_name, lncRNA_name, miRNA_sequence, lncRNA_sequence, label])
    column_names = ['miRNA_name', 'lncRNA_name', 'miRNA_sequence', 'lncRNA_sequence', 'Label']
    df = pd.DataFrame(data, columns=column_names)
    unique_miRNAs = df['miRNA_name'].unique()
    unique_lncRNAs = df['lncRNA_name'].unique()
    miRNA_to_index = {miRNA: index for index, miRNA in enumerate(unique_miRNAs)}
    lncRNA_to_index = {lncRNA: index for index, lncRNA in enumerate(unique_lncRNAs)}
    return df, miRNA_to_index, lncRNA_to_index


def lnc_indicesTensor(df, lncRNA_to_index):
    indices = df['lncRNA_name'].apply(lambda x: lncRNA_to_index[x]).tolist()
    tensor = torch.tensor(indices, dtype=torch.long)
    return tensor


def mi_indicesTensor(df, miRNA_to_index):
    indices = df['miRNA_name'].apply(lambda x: miRNA_to_index[x]).tolist()
    tensor = torch.tensor(indices, dtype=torch.long)
    return tensor


def parse_txt(txt_file):
    sequences = []
    labels = []
    strip_label_count = 0
    with open(txt_file, 'r') as file:
        for line in file:
            raw_label = line.strip().split(',')[-1]
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) != 5:
                continue
            miRNAname = parts[0]
            lncRNAname = parts[1]
            miRNAseq = parts[2]
            lncRNAseq = parts[3]
            label_str = parts[4]
            try:
                if label_str != raw_label:
                    strip_label_count += 1
                label = int(label_str)
            except ValueError:
                continue
            sequences.append((miRNAname, lncRNAname, miRNAseq, lncRNAseq))
            labels.append(label)
    return sequences, labels


def normalize_tensor(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def z_score_normalize(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return (features - mean) / std


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

