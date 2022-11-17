from typing import Optional, Any, Callable, Sequence
from .._typing import DataFrame, Series

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ..base import BaseReporter, TrainedDecompositionProtocol
from ..plot import histplot, kdeplot, barplot_color_by_sign, scatterplot


__all__ = [
    "DecompositionReporter",
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
        BaseReporter.__init__(self)
        self.model = model
        self.start_id_from_first = start_id_from_first
        self.feature_name_map = feature_name_map
        self.most_n = most_n
        self.tight_layout = tight_layout
        self.bar_kw = bar_kw
        self.hist_kw = hist_kw
        self.kde_kw = kde_kw
        self.scatter_kw = scatter_kw

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

    def show(
        self,
        name: str,
        X: Optional[DataFrame] = None,
        y: Optional[Series] = None,
        target_decomp_axes: str | int | list[int] = "all",
        y_order: Optional[Sequence[str] | Callable] = None,
        **kwargs
    ):
        """Plot and show some figures to understand the given data using decomposition."""

        fig_kw = kwargs or dict()

        showing_idx = _get_showing_decomp_axis_idx(self, target_decomp_axes)
        ncols = 2

        if y is not None:
            y = y.astype("category")

        if name == "comp":
            nrows, remainder = divmod(len(showing_idx), ncols)
            if remainder > 0:
                nrows += 1

            fig_kw.setdefault("figsize", _get_figsize(ncols, nrows))
            fig_kw.setdefault("sharex", True)
            fig, ax_arr = plt.subplots(nrows, ncols, **fig_kw)

            ax_arr = ax_arr.ravel()
            df_comp = self._listup_axis_comp()

            for i in np.arange(ncols * nrows):
                ax = ax_arr[i]
                try:
                    id_ = showing_idx[i]
                except KeyError:
                    ax.axis("off")
                    break

                df_comp_curr = df_comp.query("axis_id == @id_")
                df_comp_curr = df_comp_curr.sort_values("value", ascending=False, key=np.abs).iloc[: self.most_n]

                barplot_color_by_sign(ax, df_comp_curr, "value", title=f"Component: (axis: {id_})")
                ax.set_xlim(-1.05, 1.05)
        elif name == "dist":
            if (X is None) or (y is None):
                raise ValueError("Missing X or y in args")

            nrows = len(showing_idx)

            fig_kw.setdefault("figsize", _get_figsize(ncols, nrows))
            fig_kw.setdefault("sharex", "row")
            fig, ax_arr = plt.subplots(nrows, ncols, **fig_kw)

            df_proj = self._listup_proj(X, y)
            hist_kw = self.hist_kw or dict()
            kde_kw = self.kde_kw or dict()
            hue = y.name

            if y_order is None:
                y_order = sorted(y.unique())
            elif callable(y_order):
                y_order = sorted(y.unique(), key=y_order)

            for i in np.arange(nrows):
                ax1, ax2 = ax_arr[i, :]
                id_ = showing_idx[i]

                df_proj_curr = df_proj.query("axis_id == @id_")
                histplot(ax1, df_proj_curr, "value", title=f"Histgram: (axis: {id_})", hue=hue, hue_order=y_order, **hist_kw)
                kdeplot(ax2, df_proj_curr, "value", title=f"Distribution: (axis: {id_})", hue=hue, hue_order=y_order, **kde_kw)
        elif name == "scatter":
            if (X is None) or (y is None):
                raise ValueError("Missing X or y in args")

            nrows, remainder = divmod(len(showing_idx), ncols)
            if remainder > 0:
                nrows += 1

            fig_kw.setdefault("figsize", _get_figsize(ncols, nrows, square=True))
            fig_kw.setdefault("sharex", True)
            fig, ax_arr = plt.subplots(nrows, ncols, **fig_kw)

            ax_arr = ax_arr.ravel()
            df_proj = self._listup_proj(X, y)
            scatter_kw = self.scatter_kw or dict()

            for i in np.arange(ncols * nrows):
                ax = ax_arr[i]
                try:
                    id_ = showing_idx[i]
                except KeyError:
                    ax.axis("off")
                    break

                df_proj_curr = df_proj.query("axis_id == @id_")
                scatterplot(ax, df_proj_curr, "value", y.name, title=f"Scatter Plot: (axis: {id_})", **scatter_kw)
        else:
            raise ValueError(f"Invalid name: {name}")

        if self.tight_layout:
            plt.tight_layout()
        plt.show()
        plt.close(fig)

    def compare(
        self,
        name: str,
        X1: DataFrame,
        X2: DataFrame,
        dataset_names: Optional[tuple[str, str]] = None,
        target_decomp_axes: str | int | list[int] = "all",
        **kwargs
    ):
        dataset_names = dataset_names or ("1", "2")
        y_name = "__dataset__"
        y1_seri = pd.Series([dataset_names[0]]*len(X1), name=y_name)
        y2_seri = pd.Series([dataset_names[1]]*len(X2), name=y_name)

        X_all = pd.concat((X1, X2), axis="index", ignore_index=True)
        y_all = pd.concat((y1_seri, y2_seri), axis="index", ignore_index=True)

        self.show(name, X_all, y_all, target_decomp_axes, y_order=dataset_names, **kwargs)

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
