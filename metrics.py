from dataclasses import dataclass
import sklearn.metrics as skm
from utils import safe_divide
import numpy as np
    

def fbeta_score(p, r, beta=1):
    if p == r == 0:
        return 0
    return safe_divide(1 + beta**2, (1/p) + (beta**2/r))

def classification_scalar_metrics(cm):
    assert cm.shape == (2,2)
    acc = safe_divide(cm.diagonal().sum(), cm.sum())
    se, sp   = safe_divide(cm.diagonal(), cm.sum(1))
    ppv, npv = safe_divide(cm.diagonal(), cm.sum(0))
    f1_score = fbeta_score(ppv, se)

    return {
        'acc': acc,
        'se': se,
        'sp': sp,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1_score
    }

def classification_curve_metrics(y_true, y_score):
    fpr, tpr, _ = skm.roc_curve(y_true, y_score)
    precision, recall, _ = skm.precision_recall_curve(y_true, y_score)
    roc_auc = skm.roc_auc_score(y_true, y_score)
    pr_auc = np.trapz(recall, precision)

    return {
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_curve': {
            'precision': precision,
            'recall': recall,
        },
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
    }

