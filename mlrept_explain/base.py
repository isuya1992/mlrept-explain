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

    def set_train_data(self, X: DataFrame, y: Optional[Series], copy: bool = False):
        self.X_train = X
        self.y_train = y

        if copy:
            self.X_train = self.X_train.copy()
            if self.y_train is not None:
                self.y_train = self.y_train.copy()

class TrainedDecompositionProtocol(Protocol):
    """Protocol of trained scikit-learn decomposition objects"""
    components_: ArrayLike
    n_components_: int
    feature_names_in_: list[str]

    def transform(self, X, y=None):
        ...
