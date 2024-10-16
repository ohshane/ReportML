from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import sklearn.metrics as skm
from collections import Counter


def roll(x):
    x = np.roll(x, shift=-1, axis=0)
    x = np.roll(x, shift=-1, axis=1)
    return x


def fbeta(p, r, beta=1):
    if p == r == 0:
        return 0
    return safe_divide(1 + beta**2, (1/p) + (beta**2/r))


def safe_divide(a, b):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = 0

    return result if result.size > 1 else result.item()


def clopper_pearson(x, n, alpha=0.05):
    import scipy
    import math
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi


def confusion_matrix(y_true: np.array,
                     y_pred: np.array,
                     class_map: dict):
    assert len(np.unique(y_true)) == len(class_map)
    return skm.confusion_matrix(y_true=y_true,
                                y_pred=y_pred,
                                labels=list(class_map.values()))


def binarize_confusion_matrix(cm, class_idx):
    neg1, pos, neg2 = np.vsplit(cm, (class_idx, class_idx+1))

    a, b, c = map(lambda x: x.sum(), np.hsplit(neg1, (class_idx, class_idx+1)))
    d, e, f = map(lambda x: x.sum(), np.hsplit(pos , (class_idx, class_idx+1)))
    g, h, i = map(lambda x: x.sum(), np.hsplit(neg2, (class_idx, class_idx+1)))

    TP = e
    FN = d + f
    FP = b + h
    TN = a + c + g + i

    cm2 = np.array([[TP, FN], [FP, TN]])
    return cm2

def binarized_scalar_metrics(cm2):
    acc      = safe_divide(cm2.diagonal().sum(), cm2.sum())
    se, sp   = safe_divide(cm2.diagonal(),       cm2.sum(1))
    ppv, npv = safe_divide(cm2.diagonal(),       cm2.sum(0))
    f1 = fbeta(ppv, se)

    return {'acc': acc,
            'se' : se,
            'sp' : sp,
            'ppv': npv,
            'npv': npv,
            'f1' : f1 }

class Report:
    def __new__(cls, mode: str, *args, **kwargs):
        if mode == 'classification/binary':
            instance = super().__new__(BinaryClassificationReport)
        elif mode == 'classification/multiclass':
            instance = super().__new__(MulticlassClassificationReport)
        else:
            raise ValueError(f"Invalid report type: {mode}")
        
        instance.__init__(*args, **kwargs)
        return instance


class ClassificationReport(ABC):
    def __init__(self,
                 class_map: dict[str,int],
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None):

        values = list(class_map.values())
        assert len(set(values)) == len(values)
        assert set(np.unique(y_true)) == set(values) == set(range(len(values)))
        if y_proba is not None:
            assert y_proba.shape[-1] == len(class_map)
        assert '_' not in list(class_map.keys())
        
        self.class_map = class_map
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.cm = None

    @abstractmethod
    def generate_report(self, *args, **kwargs):
        pass

    def to_csv(self, file_name, *args, **kwargs):
        raise NotImplementedError

    def to_xlsx(self, file_name, *args, **kwargs):
        raise NotImplementedError

    def binarized_metrics(self, cm, positive_idx):
        cm = binarize_confusion_matrix(cm, positive_idx)
        metrics = binarized_scalar_metrics(cm)

        y_true  = (self.y_true==positive_idx).astype(int)
        y_score = None
        if self.y_proba is not None:
            y_score = self.y_proba.T[positive_idx]
        else:
            y_score = (self.y_pred==positive_idx).astype(int)
        
        fpr, tpr, _ = skm.roc_curve(y_true=y_true,
                                    y_score=y_score)
        pre, rec, _ = skm.precision_recall_curve(y_true=y_true,
                                                 probas_pred=y_score)
        rocauc = skm.auc(fpr, tpr)
        prauc  = skm.average_precision_score(y_true, y_score)

        metrics['rocauc'] = rocauc
        metrics['prauc']  = prauc

        misc = {
            'rocauc': {'fpr': fpr, 'tpr': tpr},
            'prauc' : {'pre': pre, 'rec': rec},
        }

        return metrics, misc


class BinaryClassificationReport(ClassificationReport):
    def __init__(self, positive_idx=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_report(positive_idx)

    def generate_report(self, positive_idx):
        cm = confusion_matrix(self.y_true,
                              self.y_pred,
                              self.class_map)

        self.cm = {'_': cm}

        metrics, misc = self.binarized_metrics(cm, positive_idx)

        self.metrics = {'_': metrics}
        self.misc = {'_': misc}
    
    def to_csv(self, file_name):
        raise NotImplementedError
    
    def to_xlsx(self, file_name):
        raise NotImplementedError


class MulticlassClassificationReport(ClassificationReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_report()

    def generate_report(self):
        cm = confusion_matrix(self.y_true,
                              self.y_pred,
                              self.class_map)

        self.cm = {'_': cm}
        self.metrics = {}
        self.misc = {}

        cm2s = list(map(lambda x: binarize_confusion_matrix(cm, x), self.class_map.values()))
        w = np.array(list(map(lambda v: (self.y_true == v).sum(),
                              self.class_map.values()))).astype(float)
        w /= w.sum()

        cm2s = np.stack(cm2s)
        metrics, misc = zip(*map(lambda x: self.binarized_metrics(cm2s[x], x),
                                 self.class_map.values()))
        metrics = dict(zip(self.class_map.keys(), metrics))
        misc    = dict(zip(self.class_map.keys(), misc))

        self.cm.update(dict(zip(self.class_map.keys(), cm2s)))
        self.metrics = metrics
        self.misc = misc
    
    def to_csv(self, file_name):
        raise NotImplementedError
    
    def to_xlsx(self, file_name):
        print(self.metrics)
        print(file_name)