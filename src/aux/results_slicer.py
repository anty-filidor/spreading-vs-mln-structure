"""Script with n auxiliary class to slice spreading results."""

from itertools import product
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd


class ResultsSlicer:

    def __init__(
        self,
        results_paths: list[str],
        baseline_type: str,
        with_repetition: bool = False
    ) -> None:
        self.raw_df = self.read_raw_df(results_paths, with_repetition)
        self.protocols = self.raw_df["protocol"].unique().tolist()
        self.mi_values = self.raw_df["mi_value"].unique().tolist()
        self.seed_budgets = self.raw_df["seed_budget"].unique().tolist()
        self.ss_methods = self.raw_df["ss_method"].unique().tolist()
        self.baseline_type = baseline_type

    @staticmethod
    def _area_under_curve(row: pd.Series) -> float:
        """Compute AuC from cdf in range [0,1] and discarded seed set impact."""
        cdf = np.cumsum(row["expositions_rec"], axis=0)
        if len(cdf) < 2:
            return np.nan
        start_value = row["seed_nb"]
        max_vaue = row["exposed_nb"] + row["unexposed_nb"]
        cdf_scaled = (cdf - start_value) / (max_vaue - start_value)
        cdf_steps = np.linspace(0, 1, len(cdf_scaled))
        return float(np.trapezoid(cdf_scaled, cdf_steps))

    @staticmethod
    def _split_str(row: pd.Series, sep: str, idx: int) -> str:
        """Split field `network` from the series and obtain its nth element."""
        row_splitted = row["network"].split(sep)
        if row_splitted[0] == row["network"]:
            return row["network"]
        elif idx >= len(row_splitted):
            raise ValueError
        return row_splitted[idx]

    def read_raw_df(self, raw_result_paths: list[str], with_repetition: bool) -> pd.DataFrame:
        dfs = []
        for csv_path in raw_result_paths:
            csv_df = pd.read_csv(csv_path)
            csv_df["expositions_rec"] = csv_df["expositions_rec"].map(
                lambda x: [int(xx) for xx in x.split(";")]
            )
            csv_df["net_type"] = csv_df.apply(self._split_str, sep="-", idx=0, axis=1)
            csv_df["net_name"] = csv_df.apply(self._split_str, sep="-", idx=1, axis=1)
            csv_df["auc"] = csv_df.apply(self._area_under_curve, axis=1)
            if with_repetition:
                csv_df["repetition"] = Path(csv_path).stem.split("_")[-1]
            dfs.append(csv_df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def get_combinations(self) -> Generator[tuple[str, str, str], None, None]:
        for and_case in product(
            sorted(self.protocols),
            sorted(self.mi_values),
            sorted(self.seed_budgets),
            sorted(self.ss_methods),
        ):
            yield and_case
    
    def get_net_types(self) -> Generator[tuple[str, str, str], None, None]:
        for net_type in sorted(self.raw_df["net_type"].unique()):
            yield net_type

    def get_net_names(self, net_type: str) -> Generator[tuple[str, str, str], None, None]:
        for net_name in sorted(
            self.raw_df.loc[self.raw_df["net_type"] == net_type ]["net_name"].unique()
        ):
            yield net_name

    def get_slice(
        self,
        protocol: str,
        mi_value: float,
        seed_budget: int,
        ss_method: str,
        net_type: str,
        net_name: str | None = None,
    ) -> pd.DataFrame:
        slice_df = self.raw_df.loc[
            (self.raw_df["protocol"] == protocol) &
            (self.raw_df["mi_value"] == mi_value) &
            (self.raw_df["seed_budget"] == seed_budget) &
            (self.raw_df["ss_method"] == ss_method) &
            (self.raw_df["net_type"] == net_type)
        ].copy()
        if net_name:
            slice_df = slice_df.loc[(self.raw_df["net_name"] == net_name)]
        return slice_df.reindex()

    @staticmethod
    def get_actors_nb(slice_df: np.ndarray) -> np.ndarray:
        return (slice_df.iloc[0]["exposed_nb"] + slice_df.iloc[0]["unexposed_nb"]).astype(int).item()

    def mean_expositions_rec(self, slice_df: pd.DataFrame) -> dict[str, np.array]:
        max_len = max(slice_df["expositions_rec"].map(lambda x: len(x)))
        exp_recs_padded = np.zeros([len(slice_df), max_len])
        for run_idx, step_idx in enumerate(slice_df["expositions_rec"]):
            exp_recs_padded[run_idx][0:len(step_idx)] = step_idx
        return {
            "actors_nb": self.get_actors_nb(slice_df),
            "avg": np.mean(exp_recs_padded, axis=0).round(3),
            "std": np.std(exp_recs_padded, axis=0).round(3),
            "cdf": np.cumsum(np.mean(exp_recs_padded, axis=0)).round(3),
        }
