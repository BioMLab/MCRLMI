import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sk_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score


def metrics1(y_true, y_score):
    auc = sk_metrics.roc_auc_score(y_true, y_score)
    precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_score)
    au_prc = sk_metrics.auc(recall, precision)
    y_pred = [0 if i < 0.5 else 1 for i in y_score]
    acc = sk_metrics.accuracy_score(y_true, y_pred)
    pre = sk_metrics.precision_score(y_true, y_pred)
    rec = sk_metrics.recall_score(y_true, y_pred)
    f1 = sk_metrics.f1_score(y_true, y_pred)
    return {auc, au_prc, acc, rec, pre, f1}


def metrics2(targets, outputs):
    auc = roc_auc_score(targets, outputs)
    aupr = average_precision_score(targets, outputs)
    binary_outputs = np.where(np.array(outputs) > 0.5, 1, 0)
    accuracy = accuracy_score(targets, binary_outputs)
    recall = recall_score(targets, binary_outputs)
    precision = precision_score(targets, binary_outputs)
    f1 = f1_score(targets, binary_outputs)
    return {'AUC':auc, 'AUPR':aupr, 'Acc':accuracy, 'Rec':recall, 'Pre':precision, 'F1':f1}


def metrics_util1(outputs, targets):
    auc = roc_auc_score(targets, outputs)
    auprc = average_precision_score(targets, outputs)
    preds = (outputs > 0.5).astype(int)
    acc = accuracy_score(targets, preds)
    pre = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    return auc, auprc, acc, rec, pre, f1

