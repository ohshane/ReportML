import numpy as np
import re
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


def excluder(d, exclude_regex=None, tolist=True):
    if exclude_regex is None:
        return d

    result = {}
    for k, v in d.items():
        if re.search(exclude_regex, k):
            continue

        if isinstance(v, dict):
            result[k] = excluder(v, exclude_regex)
        else:
            if isinstance(v, np.ndarray) and tolist:
                v = v.tolist()
            result[k] = v
    return result


def flatten(d, parent_key='', sep='.', tolist=True):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            if isinstance(v, np.ndarray) and tolist:
                v = v.tolist()
            items.append((new_key, v))
    
    return dict(items)
