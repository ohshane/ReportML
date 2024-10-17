import numpy as np
import sklearn.metrics as skm


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


def roll(x):
    x = np.roll(x, shift=-1, axis=0)
    x = np.roll(x, shift=-1, axis=1)
    return x


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

    return np.array([[TP, FN],
                     [FP, TN]])
    
