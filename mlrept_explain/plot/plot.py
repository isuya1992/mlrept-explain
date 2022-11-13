from typing import Optional, Sequence
from .._typing import Axes, DataFrame

import numpy as np

import seaborn as sns


__all__ = [
    "histplot",
    "kdeplot",
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
