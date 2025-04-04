"""Script with n auxiliary class to slice spreading results."""

from collections import Counter
from pathlib import Path

import network_diffusion as nd
import numpy as np
import pandas as pd


class ResultsSlicer:

    def __init__(self, raw_results_path: str, with_repetition: bool = False) -> None:
        self.raw_df = self.read_raw_df(raw_results_path, with_repetition)

    @staticmethod
    def read_raw_df(raw_result_paths: list[str], with_repetition: bool) -> pd.DataFrame:
        dfs = []
        print(set(Path(csv_path).parent.name for csv_path in raw_result_paths))
        for csv_path in raw_result_paths:
            csv_df = pd.read_csv(csv_path)
            if with_repetition:
                csv_df["repetition"] = Path(csv_path).stem.split("_")[-1]
            dfs.append(csv_df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def get_slice(
        self, protocol: str, mi_value: float, seed_budget: int, network:str, ss_method: str
    ) -> pd.DataFrame:
        slice_df = self.raw_df.loc[
            (self.raw_df["protocol"] == protocol) &
            (self.raw_df["mi_value"] == mi_value) &
            (self.raw_df["seed_budget"] == seed_budget) &
            (self.raw_df["network"] == network) &
            (self.raw_df["ss_method"] == ss_method)
        ].copy()
        slice_df["expositions_rec"] = slice_df["expositions_rec"].map(lambda x: [int(xx) for xx in x.split(";")])
        return slice_df.reindex()

    @staticmethod
    def mean_expositions_rec(slice_df: pd.DataFrame) -> dict[str, np.array]:
        max_len = max(slice_df["expositions_rec"].map(lambda x: len(x)))
        exp_recs_padded = np.zeros([len(slice_df), max_len])
        for run_idx, step_idx in enumerate(slice_df["expositions_rec"]):
            exp_recs_padded[run_idx][0:len(step_idx)] = step_idx
        return {
            "avg": np.mean(exp_recs_padded, axis=0).round(3),
            "std": np.std(exp_recs_padded, axis=0).round(3),
            "cdf": np.cumsum(np.mean(exp_recs_padded, axis=0)).round(3),
        }

    @staticmethod
    def get_seeds_with_frequency(slice_df: pd.DataFrame) -> tuple[list[str], list[float]]:
        seed_sets = slice_df["seed_ids"].map(lambda x: x.split(";")).to_list()
        seed_sets = np.array(seed_sets).flatten()
        seeds_counted = Counter(seed_sets)
        seed_ids = [str(key) for key in list(seeds_counted.keys())]
        seed_frequency = np.array(list(seeds_counted.values())) / sum(seeds_counted.values())
        # return seed_ids, np.log10(seed_frequency * 100).clip(0, 1)
        return seed_ids, (seed_frequency * 5).clip(0, 1)
        # return seed_ids, seed_frequency

    @staticmethod
    def get_actors_nb(slice_df: np.ndarray) -> np.ndarray:
        return (slice_df.iloc[0]["exposed_nb"] + slice_df.iloc[0]["unexposed_nb"]).astype(int).item()

    def obtain_seed_sets_for_simulated_case(
        self,
        raw_df: pd.DataFrame,
        network: str,
        protocol: str,
        seed_budget: int,
        mi_value: float,
        ss_method: str
    ) -> list[set[str]]:
        seed_sets = raw_df.loc[
            (self.raw_df["network"] == network) &
            (self.raw_df["protocol"] == protocol) &
            (self.raw_df["seed_budget"] == seed_budget) &
            (self.raw_df["mi_value"] == mi_value) &
            (self.raw_df["ss_method"] == ss_method)
        ]["seed_ids"].to_list()
        return [set(seed_set.split(";")) for seed_set in seed_sets]

    @staticmethod
    def prepare_centrality(net: nd.MultilayerNetwork, centrality: str) -> tuple[dict[str, int], dict[int, int]]:
        if centrality == "degree":
            func = nd.mln.functions.degree
            denom = sum(net.get_nodes_num().values())
        elif centrality == "neighbourhood_size":
            func = nd.mln.functions.neighbourhood_size
            denom = net.get_actors_num()
        else:
            raise AttributeError("Unsupported centrality function!")
        centrality_dict = {str(actor.actor_id): value for actor, value in func(net).items()}
        histogram_cardinal = dict(Counter(list(centrality_dict.values())))
        return centrality_dict, {key: value / denom for key, value in histogram_cardinal.items()}

