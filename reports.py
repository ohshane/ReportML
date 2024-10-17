from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

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
                 y_true   : np.ndarray,
                 y_pred   : np.ndarray,
                 y_proba  : Optional[np.ndarray] = None):

        values = list(class_map.values())
        assert len(set(values)) == len(values)
        assert set(np.unique(y_true)) == set(values) == set(range(len(values)))
        if y_proba is not None:
            assert y_proba.shape[-1] == len(class_map)
        
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


class BinaryClassificationReport(ClassificationReport):
    def __init__(self, positive_idx=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_report(positive_idx)

    def generate_report(self, positive_idx):
        raise NotImplementedError


class MulticlassClassificationReport(ClassificationReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_report()

    def generate_report(self):
        raise NotImplementedError
