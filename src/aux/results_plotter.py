"""Script with n auxiliary class to slice spreading results."""

from itertools import product
from typing import Generator

import matplotlib
import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt

from src.aux import NML_ACTORS_COLOUR, MDS_ACTORS_COLOUR, OTHER_ACTORS_COLOUR


class ResultsPlotter:

    _protocol_and = "AND"
    _protocol_or = "OR"
    _seed_budgets_and = [15, 20, 25, 30, 35]
    _seed_budgets_or = [5, 10, 15, 20, 25]
    _mi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _ss_methods = [
        "deg_c",
        "deg_cd",
        "nghb_1s",
        "nghb_sd",
        "random",
    ]
    _networks_groups = {
        "aucs": "real",
        "ckm_physicians": "real",
        # "er1": "er",
        "er2": "er",
        "er3": "er",
        "er5": "er",
        "lazega": "real",
        "l2_course_net_1": "real",
        # "sf1": "sf",
        "sf2": "sf",
        "sf3": "sf",
        "sf5": "sf",
        "timik1q2009": "real",
    }
    _networks = list(_networks_groups.keys())
    _centralities = {
        "deg_c": "degree",
        "deg_cd": "degree",
        "nghb_1s": "neighbourhood_size",
        "nghb_sd": "neighbourhood_size",
        "random": "degree",
    }

    def yield_page(self) -> Generator[tuple[str, str, str], None, None]:
        for and_case in product(
            self._networks,
            [self._protocol_and],
            self._ss_methods,
        ):
            yield and_case
        for or_case in product(
            self._networks,
            [self._protocol_or],
            self._ss_methods,
        ):
            yield or_case
    
    def yield_heatmap_config(self) -> Generator[tuple[str, str, tuple[str, str]], None, None]:
        for and_case in product(
            self._ss_methods,
            [self._protocol_and],
            list(self._networks_groups.items()),
        ):
            yield and_case
        for or_case in product(
            self._ss_methods,
            [self._protocol_or],
            list(self._networks_groups.items()),
        ):
            yield or_case
    
    def yield_figure(self, protocol: str) -> Generator[tuple[int, float], None, None]:
        if protocol == "AND":
            for and_case in product(
                self._seed_budgets_and,
                self._mi_values,
            ):
                yield and_case
        elif protocol == "OR":
            for or_case in product(
                self._seed_budgets_or,
                self._mi_values,
            ):
                yield or_case
        else:
            raise AttributeError(f"Unknown protocol {protocol}!")

    @staticmethod
    def plot_avg_with_std(record: list[dict[str, float]], ax: matplotlib.axes.Axes, label: str, colour: str):
        y_avg = record["cdf"]
        y_std = record["std"]
        x = np.arange(len(y_avg))
        ax.plot(x, y_avg, label=label, color=colour)
        ax.fill_between(x, y_avg - y_std, y_avg + y_std, color=colour, alpha=0.4)

    def plot_single_comparison_dynamics(
        self,
        record_mds: list[dict[str, float]],
        record_nml: list[dict[str, float]],
        actors_nb: int,
        mi_value: float,
        seed_budget: int,
        ax: matplotlib.axes.Axes,
    ) -> None:
        plt.rc("legend", fontsize=8)
        x_max = max(len(record_mds["avg"]), len(record_nml["avg"])) - 1
        self.plot_avg_with_std(record_mds, ax, "MDS", MDS_ACTORS_COLOUR)
        self.plot_avg_with_std(record_nml, ax, "NML", NML_ACTORS_COLOUR)
        ax.hlines(y=actors_nb, xmin=0, xmax=x_max, color="red")
        # ax.set_xlabel("Step")
        # ax.set_ylabel("Expositions")
        ax.set_xlim(left=0, right=x_max)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_visible(False)
        ax.legend(loc="lower right")
        ax.set_title(f"mu={mi_value}, |S|={seed_budget}")

    @staticmethod
    def plot_dummy_fig(mi_value: float, seed_budget: int, ax: matplotlib.axes.Axes) -> None:
        ax.set_title(f"No results for mu={mi_value}, |S|={seed_budget}")

    @staticmethod
    def plot_single_comparison_centralities(
        record_mds: list[dict[str, float]],
        record_nml: list[dict[str, float]],
        all_centralities: dict[str, int],
        hist_centralities: dict[int, int],
        mi_value: float,
        seed_budget: int,
        ax: matplotlib.axes.Axes,
    ) -> None:
        sf_mds = ResultsSlicer.get_seeds_with_frequency(record_mds)
        sf_nml = ResultsSlicer.get_seeds_with_frequency(record_nml)
        plt.rc("legend", fontsize=8)
        ymax = max(hist_centralities.values()) * 1.2
        degrees_mds = [all_centralities[seed] for seed in sf_mds[0]]
        degrees_nml = [all_centralities[seed] for seed in sf_nml[0]]
        ax.scatter(hist_centralities.keys(), hist_centralities.values(), marker=".", color=OTHER_ACTORS_COLOUR)
        ax.vlines(x=degrees_mds, ymin=0, ymax=ymax/2, label="MDS", colors=MDS_ACTORS_COLOUR, alpha=sf_mds[1])
        ax.vlines(x=degrees_nml, ymin=ymax/2, ymax=ymax, label="NML", colors=NML_ACTORS_COLOUR, alpha=sf_nml[1])
        ax.set_xlim(left=0, auto=True)
        ax.set_ylim(bottom=0, top=ymax, auto=True)
        ax.yaxis.set_visible(False)
        ax.legend(loc="upper right")
        ax.set_title(f"mu={mi_value}, |S|={seed_budget}")
