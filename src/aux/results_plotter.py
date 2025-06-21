"""Script with an auxiliary class to plot results."""

from typing import Any

import matplotlib
import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from src.aux import BASELINE_ACTORS_COLOUR, BASELINE_ACTORS_LINE


class ResultsPlotter:

    @staticmethod
    def plot_avg_with_std(
        record: dict[str, float],
        ax: matplotlib.axes.Axes,
        label: str,
        x_max: int,
        colour: str | None = None,
        shape: str | None = None,
    ) -> None:
        y_avg = np.ones(x_max) * record["cdf"][-1]
        y_avg[:len(record["cdf"])] = record["cdf"]
        y_avg = y_avg / record["actors_nb"]
        y_std = np.zeros(x_max)
        y_std[:len(record["std"])] = record["std"]
        y_std = y_std / record["actors_nb"]
        x = np.arange(x_max)
        curve = ax.plot(x, y_avg, label=label, color=colour, linestyle=shape)
        ax.fill_between(x, y_avg - y_std, y_avg + y_std, color=curve[0].get_color(), alpha=0.25)

    def plot_single_comparison_dynamics(
        self,
        records_experiments: dict[str, dict[str, Any]],
        baseline_key: str,
        title: str,
        ax: Axes,
    ) -> None:
        plt.rc("legend", fontsize=8)
        x_max = max([len(re["avg"]) for re in records_experiments.values()])
        for re_name, re_curve in records_experiments.items():
            if baseline_key == re_name:
                colour = BASELINE_ACTORS_COLOUR
                shape = BASELINE_ACTORS_LINE
            else:
                colour = None
                shape = None
            self.plot_avg_with_std(re_curve, ax, re_name, x_max, colour=colour, shape=shape)
        ax.set_xlabel("Step")
        ax.set_ylabel("Expositions")
        ax.set_xlim(left=0, right=x_max - 1)
        ax.set_ylim(bottom=0) #, top=1)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=False))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(loc="lower right")
        ax.set_title(title)

    @staticmethod
    def plot_dummy_fig(mi_value: float, seed_budget: int, ax: matplotlib.axes.Axes) -> None:
        ax.set_title(f"No results for mu={mi_value}, |S|={seed_budget}")
