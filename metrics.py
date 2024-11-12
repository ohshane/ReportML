import math
import sklearn.metrics as skm
from utils import safe_divide, clopper_pearson
import numpy as np
    

def fbeta_score(p, r, beta=1):
    if p == r == 0:
        return 0
    return safe_divide(1 + beta**2, (1/p) + (beta**2/r))


def classification_scalar_metrics(cm, ci):
    assert cm.shape == (2,2)

    acc_x, acc_n = cm.diagonal().sum(), cm.sum()
    acc = safe_divide(acc_x, acc_n)

    se_x, sp_x = cm.diagonal()
    se_n, sp_n = cm.sum(1)
    se = safe_divide(se_x, se_n)
    sp = safe_divide(sp_x, sp_n)

    ppv_x, npv_x = cm.diagonal()
    ppv_n, npv_n = cm.sum(0)
    ppv = safe_divide(ppv_x, ppv_n)
    npv = safe_divide(npv_x, npv_n)

    f1_score = fbeta_score(ppv, se)

    if ci:
        acc_ci = (acc_lo, acc_hi) = clopper_pearson(acc_x, acc_n)
        se_ci  = ( se_lo,  se_hi) = clopper_pearson( se_x,  se_n)
        sp_ci  = ( sp_lo,  sp_hi) = clopper_pearson( sp_x,  sp_n)
        ppv_ci = (ppv_lo, ppv_hi) = clopper_pearson(ppv_x, ppv_n)
        npv_ci = (npv_lo, npv_hi) = clopper_pearson(npv_x, npv_n)

        f1_score_lo = fbeta_score(ppv_lo, se_lo)
        f1_score_hi = fbeta_score(ppv_hi, se_hi)
        f1_score_ci = (f1_score_lo, f1_score_hi)

        return {
            'acc': {
                'value': acc,
                'ci': dict(zip(['lo','hi'], acc_ci)),
            },
            'se': {
                'value': se,
                'ci': dict(zip(['lo','hi'], se_ci)),
            },
            'sp': {
                'value': sp,
                'ci': dict(zip(['lo','hi'], sp_ci)),
            },
            'ppv': {
                'value': ppv,
                'ci': dict(zip(['lo','hi'], ppv_ci)),
            },
            'npv': {
                'value': npv,
                'ci': dict(zip(['lo','hi'], npv_ci)),
            },
            'f1_score':{
                'value': f1_score,
                'ci': dict(zip(['lo','hi'], f1_score_ci)),
            },
        }

    return {
        'acc': acc,
        'se': se,
        'sp': sp,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1_score
    }


def classification_curve_metrics(y_true, y_score, ci):
    fpr, tpr, _ = skm.roc_curve(y_true, y_score)
    precision, recall, _ = skm.precision_recall_curve(y_true, y_score)
    roc_auc = skm.roc_auc_score(y_true, y_score)
    pr_auc = np.trapz(recall, precision)
    
    roc_auc_n = len(y_true)
    roc_auc_x = math.floor(roc_auc_n * roc_auc)
    roc_auc_ci = (roc_auc_lo, roc_auc_hi) = clopper_pearson(roc_auc_x, roc_auc_n)

    pr_auc_n = len(y_true)
    pr_auc_x = math.floor(pr_auc_n * pr_auc)
    pr_auc_ci = (pr_auc_lo, pr_auc_hi) = clopper_pearson(pr_auc_x, pr_auc_n)

    # # TODO: metric pr auc
    # cm = skm.confusion_matrix(y_true, y_score > 0.5)
    # _, p = cm.diagonal() / cm.sum(0)
    # _, r = cm.diagonal() / cm.sum(1)
    # p_ci = (p_lo, p_hi) = clopper_pearson(p, cm.sum(0))
    # r_ci = (r_lo, r_hi) = clopper_pearson(r, cm.sum(1))

    if ci:
        return {
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
            },
            'pr_curve': {
                'precision': precision,
                'recall': recall,
            },
            'roc_auc': {
                'value': roc_auc,
                'ci': dict(zip(['lo','hi'], roc_auc_ci)),
            },
            'pr_auc': {
                'value': pr_auc,
                'ci': dict(zip(['lo','hi'], pr_auc_ci)),
            },
        }


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

