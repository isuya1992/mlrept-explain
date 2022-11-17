from typing import Protocol, Optional
from numpy.typing import ArrayLike
from ._typing import DataFrame, Series

from abc import abstractmethod
from abc import ABCMeta


__all__ = [
    "BaseReporter",
    "TrainedDecompositionProtocol",
]


class BaseReporter(metaclass=ABCMeta):
    @abstractmethod
    def show(self, name: str, X: DataFrame, y: Optional[Series], **kwargs):
        """Abstract method which offers visualization of data"""
        ...

    def fit(self, X: DataFrame, y: Series):
        self.X_train = X.copy()
        self.y_train = y.copy()

class TrainedDecompositionProtocol(Protocol):
    """Protocol of trained scikit-learn decomposition objects"""
    components_: ArrayLike
    n_components_: int
    feature_names_in_: list[str]

    def fit(self, X, y=None):
        ...

    def transform(self, X, y=None):
        ...
