from typing import Protocol
from numpy.typing import ArrayLike

from abc import abstractmethod
from abc import ABCMeta


__all__ = [
    "BaseReporter",
    "TrainedDecompositionProtocol",
]


class BaseReporter(metaclass=ABCMeta):
    @abstractmethod
    def show(self, name: str, **kwargs):
        """Abstract method which offers visualization of data"""
        ...


class TrainedDecompositionProtocol(Protocol):
    """Protocol of trained scikit-learn decomposition objects"""
    components_: ArrayLike
    n_components_: int
    feature_names_in_: list[str]

    def fit(self, X, y=None):
        ...

    def transform(self, X, y=None):
        ...
