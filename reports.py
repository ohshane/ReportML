from abc import ABC
import sklearn.metrics as skm
import numpy as np
from utils import binarize_confusion_matrix, scalar_metrics, curve_metrics
from pprint import pprint
from metrics import ClassificationMetrics, CurveMetrics

class Report:
    def __new__(cls, mode: str, *args, **kwargs):
        if mode == 'classification/binary':
            instance = super().__new__(BinaryClassificationReport)
        elif mode == 'classification/multiclass':
            instance = super().__new__(MulticlassClassificationReport)
        else:
            raise NotImplementedError
        
        instance.__init__(*args, **kwargs)
        return instance

class ClassificationReport(ABC):
    def __init__(self,
                 class_map: dict[str,int],
                 y_true   : np.ndarray,
                 y_pred   : np.ndarray,
                 y_proba  : np.ndarray = None):

        values = list(class_map.values())
        assert len(set(values)) == len(values)
        assert set(np.unique(y_true)) == set(values) == set(range(len(values)))
        if y_proba is not None:
            assert y_proba.shape[-1] == len(class_map)
        
        self.class_map = class_map
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def generate_metrics(self, *args, **kwargs):
        raise NotImplementedError

    def to_csv(self, file_name, *args, **kwargs):
        raise NotImplementedError

    def to_xlsx(self, file_name, *args, **kwargs):
        raise NotImplementedError


class BinaryClassificationReport(ClassificationReport):
    def __init__(self, positive_idx=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_metrics(positive_idx)

    def generate_metrics(self, positive_idx):
        cm = skm.confusion_matrix(y_true=self.y_true,
                                  y_pred=self.y_pred,
                                  labels=list(self.class_map.values()))
        cm = binarize_confusion_matrix(cm, list(self.class_map.values()).index(positive_idx))
        print(cm)

        y_true = (self.y_true == positive_idx).astype(int)
        y_score = None

        if self.y_proba is None:
            y_score = (self.y_pred == positive_idx).astype(float)
        else:
            y_score = self.y_proba.T[positive_idx]
        
        self.metrics = {
            'cm': cm,
            'scalar': ClassificationMetrics(**scalar_metrics(cm)),
            'curve': CurveMetrics(**curve_metrics(y_true, y_score))
        }

class MulticlassClassificationReport(ClassificationReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_report()

    def generate_report(self):
        raise NotImplementedError
