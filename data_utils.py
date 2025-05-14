import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
import copy
import argparse
from sklearn.model_selection import StratifiedKFold
from model import *
from metrics_util import *
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.init as init


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features1, features2, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features1.is_cuda
                  else torch.device('cpu'))

        features = torch.cat((features1,features2),0)
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        batch = features1.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            mask = torch.eye(batch, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)
        mask = torch.cat((mask,mask), 1)
        mask = torch.cat((mask, mask), 0)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask).to(exp_logits.device) - torch.eye(batch_size).to(exp_logits.device)
        positives_mask = (mask * logits_mask).to(logits_mask.device)
        negatives_mask = (1. - mask).to(logits_mask.device)

        num_positives_per_row = torch.sum(positives_mask, axis=1).to(logits_mask.device)
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
