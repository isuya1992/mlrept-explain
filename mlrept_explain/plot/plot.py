from typing import Optional, Sequence
from .._typing import Axes, DataFrame

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


__all__ = [
    "histplot",
    "kdeplot",
    "scatterplot",
    "barplot",
    "barplot_color_by_sign",
]


def histplot(
    ax: Axes,
    df: DataFrame,
    x: str,
    hue: str,
    hue_order: Sequence[str],
    bins: int = 20,
    title: Optional[str] = None,
    **kwargs
):
    kwargs = kwargs or dict()
    kwargs.setdefault("histtype", "bar")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("linewidth", 0)
    kwargs.setdefault("rwidth", 0.9)

    bins_arr = np.linspace(df[x].min(), df[x].max(), bins)

    for hue_val in hue_order:
        matched_loc = df[hue].eq(hue_val)
        x_seri = df[x][matched_loc]
        ax.hist(x_seri, bins=bins_arr, label=hue_val, **kwargs)

    ax.legend(title=hue)
    ax.grid(visible=True, axis="y")
    ax.set_axisbelow(True)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("Count")

def kdeplot(
    ax: Axes,
    df: DataFrame,
    x: str,
    hue: str,
    hue_order: Sequence[str],
    title: Optional[str] = None,
    **kwargs
):
    kwargs = kwargs or dict()
    kwargs.setdefault("fill", True)

    for hue_val in hue_order:
        matched_loc = df[hue].eq(hue_val)
        x_seri = df[x][matched_loc]
        sns.kdeplot(x_seri, ax=ax, label=hue_val, **kwargs)

    ax.legend(title=hue)
    ax.grid(visible=True, axis="y")
    ax.set_axisbelow(True)
    ax.set_title(title)
    ax.set_xlabel(x)

def scatterplot(
    ax: Axes,
    df: DataFrame,
    x: str,
    y: str,
    bins: int = 20,
    title: Optional[str] = None,
    **kwargs
):
    kwargs = kwargs or dict()
    kwargs.setdefault("fill", True)

    x_seri = df[x]
    y_seri = df[y]

    ax.scatter(x_seri, y_seri)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 0.5, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 0.5, pad=0.1, sharey=ax)
    ax_histx.hist(x_seri, bins=bins, rwidth=0.9)
    ax_histy.hist(y_seri, bins=bins, rwidth=0.9, orientation='horizontal')
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    ax.grid(visible=True, axis="both")
    ax.set_axisbelow(True)
    ax.set_title(title, pad=0.7*72)  # Convert inch to point.
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def barplot(
    ax: Axes,
    df: DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    invert_yaxis: bool = True,
    **kwargs
):
    kwargs = kwargs or dict()
    kwargs.setdefault("linewidth", 0)
    kwargs.setdefault("height", 0.9)

    y_seri = df[y]
    x_seri = df[x]
    ax.barh(y_seri, width=x_seri, **kwargs)

    ax.grid(visible=True, axis="x")
    ax.set_axisbelow(True)
    ax.set_title(title)
    ax.set_xlabel(x)
    if invert_yaxis:
        ax.invert_yaxis()


def barplot_color_by_sign(
    ax: Axes,
    df: DataFrame,
    y: str,
    title: Optional[str],
    **kwargs
):
    positive_idx = df[y].ge(0)
    negative_idx = positive_idx.eq(False)

    df_pos = df.assign(value=df[y].where(positive_idx, 0))
    df_neg = df.assign(value=df[y].where(negative_idx, 0))

    barplot(ax, df_pos, y, "feature", invert_yaxis=False, color="b", **kwargs)
    barplot(ax, df_neg, y, "feature", invert_yaxis=False, color="r", **kwargs)
    ax.invert_yaxis()
    ax.set_title(title)
