import numpy as np
from reports import Report
from pprint import pprint

CLASS_MAP = {
    "apple"    : 0,
    "banana"   : 1,
    "chocolate": 2,
    "dumpling" : 3,
}

def createDummy(n_samples, class_map):
    np.random.seed(42)
    y_true = np.random.choice(len(class_map), n_samples)
    logits = np.random.randn(n_samples,len(class_map))
    y_proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y_pred = y_proba.argmax(1)
    return y_true, y_pred, y_proba

if __name__ == "__main__":
    y_true, y_pred, y_proba = createDummy(n_samples=100, class_map=CLASS_MAP)

    report = Report('classification/multiclass',
                    class_map=CLASS_MAP,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba)
    
    report.to_json('report.json')