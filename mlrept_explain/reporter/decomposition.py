from typing import Optional, Any, Sequence, Callable
from .._typing import DataFrame, Series, Figure, Axes

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ..base import BaseReporter, TrainedDecompositionProtocol
from ..plot import barplot_color_by_sign, histplot, kdeplot, scatterplot
from .._exception import NotFittedError


__all__ = [
    "DecompositionReporter",
    "DecompositionReporterForClassification",
    "DecompositionReporterForRegression",
]


def _get_figsize(ncols: int, nrows: int, square: bool = False) -> tuple[float, float]:
    defalut_figsize = plt.rcParams["figure.figsize"]
    sizex, sizey = defalut_figsize

    out_sizex = ncols * sizex
    if square:
        out_sizey = nrows * sizex
    else:
        out_sizey = nrows * sizey

    return out_sizex, out_sizey


def _get_showing_decomp_axis_idx(reporter, target_decomp_axes: str | int | list[int]) -> list[int]:
    if target_decomp_axes == "all":
        showing_axes = reporter.axis_idx()
    elif isinstance(target_decomp_axes, int):
        showing_axes = [target_decomp_axes]
    elif isinstance(target_decomp_axes, list):
        showing_axes = target_decomp_axes
    else:
        raise ValueError(f"Invalid target_decomp_axes: {target_decomp_axes}")

    return showing_axes


class DecompositionReporter(BaseReporter):
    available_plot_names = ["comp"]

    """Class which offers visualization of classification dataset by dimension reduction"""
    def __init__(
        self,
        model: TrainedDecompositionProtocol,
        start_id_from_first: bool = True,
        feature_name_map: Optional[dict[str, str]] = None,
        most_n: int = 10,
        tight_layout: bool = True,
        bar_kw: Optional[dict[str, Any]] = None,
    ):
        self.model = model
        self.start_id_from_first = start_id_from_first
        self.feature_name_map = feature_name_map
        self.most_n = most_n
        self.tight_layout = tight_layout
        self.bar_kw = bar_kw

        self.X_train: Optional[DataFrame] = None
        self.y_train: Optional[Series] = None

    @property
    def feature_names(self) -> list[str]:
        if self.feature_name_map is None:
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [
                self.feature_name_map.get(name, name)
                for name in self.model.feature_names_in_
            ]
        return feature_names

    def fit(self, X: DataFrame, y: Optional[Series], **kwargs):
        self.X_train = X.copy()
        self.model.fit(self.X_train, self.y_train, **kwargs)

    def show(
        self,
        name: str,
        target_decomp_axes: str | int | list[int] = "all",
        **kwargs
    ) -> tuple[Figure, Axes]:
        """Plot and show some figures to understand the given data using decomposition."""

        if self.X_train is None:
            raise NotFittedError("Not fitted train dataset")

        fig_kw = kwargs or dict()

        showing_idx = _get_showing_decomp_axis_idx(self, target_decomp_axes)
        ncols = 2

        if name == "comp":
            nrows, remainder = divmod(len(showing_idx), ncols)
            if remainder > 0:
                nrows += 1

            fig_kw.setdefault("figsize", _get_figsize(ncols, nrows))
            fig_kw.setdefault("sharex", True)
            fig, ax_arr = plt.subplots(nrows, ncols, **fig_kw)

            df_comp = self._listup_axis_comp()

            for i in np.arange(ncols * nrows):
                ax = ax_arr.ravel()[i]
                try:
                    id_ = showing_idx[i]
                except KeyError:
                    ax.axis("off")
                    break

                df_comp_curr = df_comp.query("axis_id == @id_")
                df_comp_curr = df_comp_curr.sort_values("value", ascending=False, key=np.abs).iloc[: self.most_n]

                barplot_color_by_sign(ax, df_comp_curr, "value", title=f"Component: (axis: {id_})")
                ax.set_xlim(-1.05, 1.05)

        else:
            raise ValueError(f"Invalid name: {name}")

        if self.tight_layout:
            plt.tight_layout()

        return fig, ax_arr

    def axis_idx(self) -> Series:
        seri = pd.Series(np.arange(0, self.model.n_components_))
        seri.name = "axis_id"
        if self.start_id_from_first:
            seri += 1

        return seri

    def _listup_proj(self, X: DataFrame, y: Optional[Series]) -> DataFrame:
        # Transform X to the decomposition space.
        axis_idx = self.axis_idx()
        X_pca = pd.DataFrame(
            self.model.transform(X),
            columns=axis_idx
        )

        # Make dataFrame.
        df = X_pca
        if y is not None:
            y = y.reset_index(drop=True)  # Avoid to fail to concat data derivated from non-sequential index.
            df = pd.concat((df, y), axis="columns")
            df = df.melt(id_vars=y.name, var_name="axis_id", value_name="value")
        else:
            df = df.melt(var_name="axis_id", value_name="value")

        return df

    def _listup_axis_comp(self) -> DataFrame:
        axis_idx = self.axis_idx()
        X_axes = pd.DataFrame(
            self.model.components_,
            columns=self.feature_names,
        )

        # Make dataFrame.
        df = X_axes
        df = pd.concat((axis_idx, df), axis="columns")
        df = df.melt(id_vars="axis_id", var_name="feature", value_name="value")

        return df


class DecompositionReporterForClassification(DecompositionReporter):
    available_plot_names = ["dist"] + DecompositionReporter.available_plot_names

    """Class which offers visualization of classification dataset by dimension reduction"""
    def __init__(
        self,
        model: TrainedDecompositionProtocol,
        start_id_from_first: bool = True,
        feature_name_map: Optional[dict[str, str]] = None,
        most_n: int = 10,
        tight_layout: bool = True,
        bar_kw: Optional[dict[str, Any]] = None,
        hist_kw: Optional[dict[str, Any]] = None,
        kde_kw: Optional[dict[str, Any]] = None,
        scatter_kw: Optional[dict[str, Any]] = None,
    ):
        DecompositionReporter.__init__(self, model, start_id_from_first, feature_name_map, most_n, tight_layout, bar_kw)
        self.hist_kw = hist_kw
        self.kde_kw = kde_kw
        self.scatter_kw = scatter_kw

    def fit(self, X: DataFrame, y: Series, **kwargs):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.model.fit(self.X_train, self.y_train, **kwargs)

    def show(
        self,
        name: str,
        target_decomp_axes: str | int | list[int] = "all",
        y_order: Optional[Sequence[str] | Callable] = None,
        **kwargs
    ) -> tuple[Figure, Axes]:
        """Plot and show some figures to understand the given data using decomposition."""

        if (self.X_train is None) or (self.y_train is None):
            raise NotFittedError("Not fitted train dataset")

        fig_kw = kwargs or dict()

        showing_idx = _get_showing_decomp_axis_idx(self, target_decomp_axes)
        ncols = 2

        if name in super().available_plot_names:
            fig, ax_arr = super().show(name, target_decomp_axes, **kwargs)
        elif name == "dist":
            nrows = len(showing_idx)

            fig_kw.setdefault("figsize", _get_figsize(ncols, nrows))
            fig_kw.setdefault("sharex", "row")
            fig, ax_arr = plt.subplots(nrows, ncols, **fig_kw)

            df_proj = self._listup_proj(self.X_train, self.y_train)
            hist_kw = self.hist_kw or dict()
            kde_kw = self.kde_kw or dict()
            hue = self.y_train.name

            if y_order is None:
                y_order = sorted(self.y_train.unique())
            elif callable(y_order):
                y_order = sorted(self.y_train.unique(), key=y_order)

            for i in np.arange(nrows):
                ax1, ax2 = ax_arr[i, :]
                id_ = showing_idx[i]

                df_proj_curr = df_proj.query("axis_id == @id_")
                histplot(ax1, df_proj_curr, "value", title=f"Histgram: (axis: {id_})", hue=hue, hue_order=y_order, **hist_kw)
                kdeplot(ax2, df_proj_curr, "value", title=f"Distribution: (axis: {id_})", hue=hue, hue_order=y_order, **kde_kw)
        else:
            raise ValueError(f"Invalid name: {name}")

        if self.tight_layout:
            plt.tight_layout()

        return fig, ax_arr

    def compare(
        self,
        name: str,
        X: DataFrame,
        dataset_name: str = "test",
        target_decomp_axes: str | int | list[int] = "all",
        **kwargs
    ) -> tuple[Figure, Axes]:

        if (self.X_train is None) or (self.y_train is None):
            raise NotFittedError("Not fitted train dataset")

        y_name = "__dataset__"
        dataset_names = ("train", dataset_name)
        y_order = dataset_names
        y_train_seri = pd.Series([dataset_names[0]]*len(self.X_train), name=y_name)
        y_test_seri = pd.Series([dataset_names[1]]*len(X), name=y_name)

        X_all = pd.concat((self.X_train, X), axis="index", ignore_index=True)
        y_all = pd.concat((y_train_seri, y_test_seri), axis="index", ignore_index=True)

        fig_kw = kwargs or dict()

        showing_idx = _get_showing_decomp_axis_idx(self, target_decomp_axes)
        ncols = 2

        if name == "dist":
            nrows = len(showing_idx)

            fig_kw.setdefault("figsize", _get_figsize(ncols, nrows))
            fig_kw.setdefault("sharex", "row")
            fig, ax_arr = plt.subplots(nrows, ncols, **fig_kw)

            df_proj = self._listup_proj(X_all, y_all)
            hist_kw = self.hist_kw or dict()
            kde_kw = self.kde_kw or dict()
            hue = y_all.name

            for i in np.arange(nrows):
                ax1, ax2 = ax_arr[i, :]
                id_ = showing_idx[i]

                df_proj_curr = df_proj.query("axis_id == @id_")
                histplot(ax1, df_proj_curr, "value", title=f"Histgram: (axis: {id_})", hue=hue, hue_order=y_order, **hist_kw)
                kdeplot(ax2, df_proj_curr, "value", title=f"Distribution: (axis: {id_})", hue=hue, hue_order=y_order, **kde_kw)
        else:
            raise ValueError(f"Invalid name: {name}")

        if self.tight_layout:
            plt.tight_layout()

        return fig, ax_arr


class DecompositionReporterForRegression(DecompositionReporter):
    available_plot_names = ["scatter"] + DecompositionReporter.available_plot_names

    """Class which offers visualization of classification dataset by dimension reduction"""
    def __init__(
        self,
        model: TrainedDecompositionProtocol,
        start_id_from_first: bool = True,
        feature_name_map: Optional[dict[str, str]] = None,
        most_n: int = 10,
        tight_layout: bool = True,
        bar_kw: Optional[dict[str, Any]] = None,
        hist_kw: Optional[dict[str, Any]] = None,
        kde_kw: Optional[dict[str, Any]] = None,
        scatter_kw: Optional[dict[str, Any]] = None,
    ):
        DecompositionReporter.__init__(self, model, start_id_from_first, feature_name_map, most_n, tight_layout, bar_kw)
        self.hist_kw = hist_kw
        self.kde_kw = kde_kw
        self.scatter_kw = scatter_kw

    def fit(self, X: DataFrame, y: Series, **kwargs):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.model.fit(self.X_train, self.y_train, **kwargs)

    def show(
        self,
        name: str,
        target_decomp_axes: str | int | list[int] = "all",
        **kwargs
    ) -> tuple[Figure, Axes]:
        """Plot and show some figures to understand the given data using decomposition."""

        if (self.X_train is None) or (self.y_train is None):
            raise NotFittedError("Not fitted train dataset")

        fig_kw = kwargs or dict()

        showing_idx = _get_showing_decomp_axis_idx(self, target_decomp_axes)
        ncols = 2

        if name in super().available_plot_names:
            fig, ax_arr = super().show(name, target_decomp_axes, **kwargs)
        elif name == "scatter":
            nrows, remainder = divmod(len(showing_idx), ncols)
            if remainder > 0:
                nrows += 1

            fig_kw.setdefault("figsize", _get_figsize(ncols, nrows, square=True))
            fig_kw.setdefault("sharex", True)
            fig, ax_arr = plt.subplots(nrows, ncols, **fig_kw)

            ax_arr = ax_arr.ravel()
            df_proj = self._listup_proj(self.X_train, self.y_train)
            scatter_kw = self.scatter_kw or dict()

            for i in np.arange(ncols * nrows):
                ax = ax_arr[i]
                try:
                    id_ = showing_idx[i]
                except KeyError:
                    ax.axis("off")
                    break

                df_proj_curr = df_proj.query("axis_id == @id_")
                scatterplot(ax, df_proj_curr, "value", self.y_train.name, title=f"Scatter Plot: (axis: {id_})", **scatter_kw)
        else:
            raise ValueError(f"Invalid name: {name}")

        if self.tight_layout:
            plt.tight_layout()

        return fig, ax_arr
