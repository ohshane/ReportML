from abc import ABC
import datetime
from pprint import PrettyPrinter
import csv
import sklearn.metrics as skm
import json
import numpy as np
from utils import binarize_confusion_matrix, excluder
from metrics import classification_scalar_metrics, classification_curve_metrics


class Report:
    def __new__(cls, mode: str, *args, **kwargs):
        if mode == 'classification/binary':
            instance = BinaryClassificationReport(*args, **kwargs)
        elif mode == 'classification/multiclass':
            instance = MulticlassOvRClassificationReport(*args, **kwargs)
        else:
            raise NotImplementedError
        return instance


class ClassificationReport(ABC):
    def __init__(self,
                 class_map: dict[str,int],
                 y_true   : np.ndarray,
                 y_pred   : np.ndarray,
                 y_proba  : np.ndarray = None,
                 ci = True,
                 *args, **kwargs):

        values = list(class_map.values())
        assert len(set(values)) == len(values)
        assert set(np.unique(y_true)) == set(values) == set(range(len(values)))
        if y_proba is not None:
            assert y_proba.shape[-1] == len(class_map)
        
        self.class_map = class_map
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.ci = ci

        self.reports = []
        self.report = None
        self.generate_report(*args, **kwargs)
    
    @property
    def index_map(self) -> dict[int,str]:
        return {v: k for k, v in self.class_map.items()} 
    
    def simple_reports(self, tolist=True, *args, **kwargs) -> list:
        return list(map(lambda x: excluder(x, exclude_regex=r"_curve$", tolist=tolist), self.reports))

    def generate_report(self, *args, **kwargs):
        raise NotImplementedError
    
    def to_object(self, *args, **kwargs):
        return self.simple_reports(tolist=False)
    
    def to_json(self, file_name, *args, **kwargs):
        with open(f"{file_name}", 'w', newline='') as f:
            json.dump(self.simple_reports(), f, indent=4)

    def to_csv(self, file_name, *args, **kwargs):
        raise NotImplementedError

    def to_xlsx(self, file_name, *args, **kwargs):
        raise NotImplementedError
    
    def __repr__(self):
        pp = PrettyPrinter(sort_dicts=False)
        return pp.pformat(self.simple_reports(tolist=True))


class BinaryClassificationReport(ClassificationReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_report(self,
                        report_type='binary_classification',
                        positive_idx=1):
        cm = skm.confusion_matrix(y_true=self.y_true,
                                  y_pred=self.y_pred,
                                  labels=list(self.class_map.values()))
        cm = binarize_confusion_matrix(cm, list(self.class_map.values()).index(positive_idx))

        y_true = (self.y_true == positive_idx).astype(int)
        y_score = None

        if self.y_proba is None:
            y_score = (self.y_pred == positive_idx).astype(float)
        else:
            y_score = self.y_proba.T[positive_idx]
        
        metrics = {}
        metrics.update(classification_scalar_metrics(cm, self.ci))
        metrics.update(classification_curve_metrics(y_true, y_score, self.ci))

        report = {
            'report_type': report_type,
            'ts': f"{datetime.datetime.now()}",
            'classes': [self.index_map[positive_idx], '_'],
            'confusion_matrix': cm,
            'metrics': metrics,
        }

        self.reports.append(report)
        self.report = report
        return report


class MulticlassClassificationReport(ClassificationReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_report(self, report_type='multiclass_classification'):
        cm = skm.confusion_matrix(y_true=self.y_true,
                                  y_pred=self.y_pred,
                                  labels=list(self.class_map.values()))
        
        report = {
            'report_type': report_type,
            'ts': f"{datetime.datetime.now()}",
            'classes': list(self.class_map.keys()),
            'confusion_matrix': cm,
            'metrics': {
                'acc': cm.diagonal().sum() / cm.sum()
            }
        }

        self.reports.append(report)
        self.report = report
        return report


class MulticlassOvRClassificationReport(MulticlassClassificationReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_report(self, report_type='multiclass_ovr_classification'):
        _report = super().generate_report()
        _cm = _report['confusion_matrix']

        for i, (k, v) in enumerate(self.class_map.items()):
            cm = binarize_confusion_matrix(_cm, i)
            BinaryClassificationReport.generate_report(self,
                                                       report_type=report_type,
                                                       positive_idx=v)
        
