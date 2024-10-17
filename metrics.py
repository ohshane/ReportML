from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


def revealScalarMetric(a, b):
    desc, a_desc, b_desc = None, None, None
    a_value, b_value = a, b
    if isinstance(a, ScalarMetric):
        a_value = a.value
        a_desc  = a.desc
    if isinstance(b, ScalarMetric):
        b_value = b.value
        b_desc  = b.desc
    if a_desc == b_desc:
        desc = a_desc
    if a_desc is not None and b_desc is None:
        desc = a_desc
    if a_desc is None and b_desc is not None:
        desc = b_desc
    return a_value, b_value, desc


class Metric(ABC):
    pass


class ScalarMetric(Metric):
    def __init__(self,
                 value: float, 
                 desc : Optional[str] = None):
        self.value = value
        self.desc = desc
    
    def item(self):
        return self.value
    
    def __str__(self):
        return self.desc if self.desc is not None else ""

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value:.4f}, desc={self.desc})"

    def __add__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        return ScalarMetric(a + b, desc)

    def __sub__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        return ScalarMetric(a - b, desc)

    def __mul__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        return ScalarMetric(a * b, desc)

    def __truediv__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return ScalarMetric(a / b, desc)

    def __floordiv__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return ScalarMetric(a // b, desc)

    def __mod__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        if b == 0:
            raise ValueError("Cannot modulo by zero")
        return ScalarMetric(a % b, desc)

    def __pow__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        return ScalarMetric(a ** b, desc)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        return ScalarMetric(b - a, desc)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        if a == 0:
            raise ValueError("Cannot divide by zero")
        return ScalarMetric(b / a, desc)

    def __rfloordiv__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        if a == 0:
            raise ValueError("Cannot divide by zero")
        return ScalarMetric(b // a, desc)

    def __rmod__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        if a == 0:
            raise ValueError("Cannot modulo by zero")
        return ScalarMetric(b % a, desc)

    def __rpow__(self, other):
        a, b, desc = revealScalarMetric(self, other)
        return ScalarMetric(b ** a, desc)

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __imod__(self, other):
        return self.__mod__(other)

    def __ipow__(self, other):
        return self.__pow__(other)

    def __eq__(self, other):
        a, b, _ = revealScalarMetric(self, other)
        return a == b

    def __ne__(self, other):
        a, b, _ = revealScalarMetric(self, other)
        return a != b

    def __lt__(self, other):
        a, b, _ = revealScalarMetric(self, other)
        return a < b

    def __le__(self, other):
        a, b, _ = revealScalarMetric(self, other)
        return a <= b

    def __gt__(self, other):
        a, b, _ = revealScalarMetric(self, other)
        return a > b

    def __ge__(self, other):
        a, b, _ = revealScalarMetric(self, other)
        return a >= b

    def __neg__(self):
        self.value = -self.value
        return self

    def __pos__(self):
        self.value = +self.value
        return self

    def __abs__(self):
        self.value = abs(self.value)
        return self

    def __invert__(self):
        self.value = ~int(self.value)
        return self


class CurveMetric(Metric):
    def __init__(self,
                 x_values: np.ndarray, 
                 y_values: np.ndarray, 
                 desc    : Optional[str] = None):
        self.x_values = x_values
        self.y_values = y_values
        self.desc = desc
    
    def item(self):
        return np.stack([self.x_values, self.y_values])


class ConfusionMatrix(Metric):
    def __init__(self,
                 cm   : np.ndarray, 
                 desc : Optional[str] = None):
        self.cm = cm
        self.desc = desc
    
    def item(self):
        return np.stack([self.x_values, self.y_values])


Acc      = lambda x: ScalarMetric(value=x, desc="Accuracy")
Se       = lambda x: ScalarMetric(value=x, desc="Sensitivity")
Sp       = lambda x: ScalarMetric(value=x, desc="Specificity")
PPV      = lambda x: ScalarMetric(value=x, desc="Positive predictive value")
NPV      = lambda x: ScalarMetric(value=x, desc="Negative predictive value")
F1_score = lambda x: ScalarMetric(value=x, desc="F1 score")
ROC_auc  = lambda x: ScalarMetric(value=x, desc="Area under ROC curve")
PR_auc   = lambda x: ScalarMetric(value=x, desc="Area under PR curve")


@dataclass
class ClassificationMetrics:
    acc      : ScalarMetric
    se       : ScalarMetric
    sp       : ScalarMetric
    ppv      : ScalarMetric
    npv      : ScalarMetric
    f1_score : ScalarMetric
    roc_auc  : ScalarMetric
    pr_auc   : ScalarMetric

