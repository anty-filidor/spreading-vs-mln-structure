from itertools import combinations
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def prepare_layer_pairs(entities: list[str]) -> list[tuple[str, str]]:
    return list(combinations(entities, 2))


def _get_col_names(raw_statistics: list[dict[tuple[str, str], float]]) -> list[str]:
    """Get names of compared entities in `raw_statistics`."""
    l_names = []
    for record in raw_statistics:
        for (l1_name, l2_name), _ in record.items():
            l_names.append(l1_name)
            l_names.append(l2_name)
    return list(set(l_names))


def create_correlation_matrix(raw_statistics: list[dict[tuple[str, str], float]]) -> pd.DataFrame:
    """Create correlation matrix that can be plotted as a heatmap."""
    col_names = _get_col_names(raw_statistics)
    matrix = pd.DataFrame(index=sorted(col_names), columns=sorted(col_names), data=np.nan)
    for record in raw_statistics:
        for (la_name, lb_name), statistic in record.items():
            matrix.loc[la_name, lb_name] = statistic
            matrix.loc[lb_name, la_name] = statistic
    for l_name in col_names:
        matrix.loc[l_name, l_name] = 1.0
    return matrix


def prepare_ticklabels(series: pd.Index) -> Union[np.ndarray, str]:
    try:
        return series.to_numpy().round(2)
    except:
        return "auto"


def plot_heatmap(
    vis_df: pd.DataFrame,
    heatmap_ax: plt.Axes,
    bar_ax: plt.Axes,
    title: str,
    vrange=(-1., 1.),
    cmap="GnBu_r", # "RdYlGn",
    mask: Optional[pd.DataFrame] = None,
    fmt: Optional[str] = ".3f",
) -> None:
    if len(vis_df.columns) >= 5:
        annot = False
        font_size = None
        yticklabels=()
        xticklabels=()
    else:
        annot = True
        font_size = 18
        yticklabels=prepare_ticklabels(vis_df.index)
        xticklabels=prepare_ticklabels(vis_df.columns)
    
    title_size = 22

    sns.heatmap(
        vis_df,
        ax=heatmap_ax,
        cbar_ax=bar_ax,
        cmap=cmap,
        vmin=vrange[0],
        vmax=vrange[1],
        annot=annot,
        annot_kws={"size": font_size},
        fmt=fmt,
        square=True,
        yticklabels=yticklabels,
        xticklabels=xticklabels,
        linewidth=.5,
        mask=mask,
        cbar= True if bar_ax is not None else False,
    )

    heatmap_ax.set_title(title, fontdict={"size": title_size})
    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontsize=font_size)
    heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), fontsize=font_size)
    # heatmap_ax.invert_yaxis()
    # heatmap_ax.tick_params(axis="x", rotation=80)
    bar_ax.tick_params(labelsize=title_size)
