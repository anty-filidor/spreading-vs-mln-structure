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
        self.probabs = self.raw_df["probab"].unique().tolist()
        self.seed_budgets = self.raw_df["seed_budget"].unique().tolist()
        self.ss_methods = self.raw_df["ss_method"].unique().tolist()
        self.baseline_type = baseline_type

    def read_raw_df(self, raw_result_paths: list[str], with_repetition: bool) -> pd.DataFrame:
        """
        for mlnabcd generated nets replace their type by a series they belong to and leave as
        a name the filename.
        """
        dfs = []
        split_func = lambda x: [int(xx) for xx in x.split(";")]
        for csv_path in raw_result_paths:
            csv_df = pd.read_csv(csv_path)
            csv_df["_network_name"] = csv_df.apply(
                lambda row: row["network_name"].split("-")[1] if row["network_type"] == "mlnabcd"
                else  row["network_name"],
                axis=1
            )
            csv_df["network_type"] = csv_df.apply(
                lambda row: row["network_name"].split("-")[0] if row["network_type"] == "mlnabcd"
                else  row["network_type"],
                axis=1
            )
            csv_df["network_name"] = csv_df["_network_name"]
            csv_df = csv_df.drop("_network_name", axis=1)
            csv_df["seed_ids"] = csv_df["seed_ids"].map(split_func)
            csv_df["expositions_rec"] = csv_df["expositions_rec"].map(split_func)
            if with_repetition:
                csv_df["repetition"] = Path(csv_path).stem.split("_")[-1]
            dfs.append(csv_df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def get_combinations(self) -> Generator[tuple[str, str, str, str], None, None]:
        for and_case in product(
            sorted(self.protocols),
            sorted(self.probabs),
            sorted(self.seed_budgets),
            sorted(self.ss_methods),
        ):
            yield and_case
    
    def get_net_types(self) -> Generator[str, None, None]:
        for net_type in sorted(self.raw_df["network_type"].unique()):
            yield net_type

    def get_net_names(self, net_type: str) -> Generator[str, None, None]:
        for net_name in sorted(
            self.raw_df.loc[self.raw_df["network_type"] == net_type ]["network_name"].unique()
        ):
            yield net_name

    def get_slice(
        self,
        protocol: str,
        probab: float | str,
        seed_budget: int | str,
        ss_method: str,
        net_type: str,
        # net_name: str,
    ) -> pd.DataFrame:
        slice_df = self.raw_df.loc[
            (self.raw_df["protocol"] == protocol) &
            (self.raw_df["probab"] == probab) &
            (self.raw_df["seed_budget"] == seed_budget) &
            (self.raw_df["ss_method"] == ss_method) &
            (self.raw_df["network_type"] == net_type)
            # (self.raw_df["network_name"] == net_name)
        ].copy()
        return slice_df.reindex()

    @staticmethod
    def get_actors_nb(slice_df: pd.DataFrame) -> float:
        return (slice_df["exposed_nb"] + slice_df["unexposed_nb"]).mean().item()

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
