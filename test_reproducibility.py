"""E2E test for runner to check reproducibly of results."""

# TODO: sth is wrong in the torch representation of the network. when actor IDs are strings, the 
# results are not 100% the same if RNG is fixed. Probably it's a foulty implementation of bidict
# used in nd.MultilayerNetworkTorch

from pathlib import Path
import os

import pandas as pd
import pytest

from src.simulator import simulate
from src.simulator.simulation_step import compute_gain, compute_area
from src.utils import set_rng_seed


@pytest.fixture
def tcase_ranking_config():
    return {
        "run": {"experiment_type": "simulate", "rng_seed": 1959, "device": "cpu"},
        "parameter_space": {
            "protocols": ["OR", "AND"],
            "probabs": [0.9, 0.65, 0.1],
            "seed_budgets": [10, 20, 30],
            "ss_methods": ["deg_c", "random"],
            "networks": ["smallreal^toy_network", "smallreal^l2_course_net_1"],
        },
        "simulator": {"max_epochs_num": 10, "repetitions": 3},
        "io": {
            "ranking_path": None,
            "compress_to_zip": False,
            "out_dir": None,
        },
    }


@pytest.fixture
def tcase_ranking_csv_names():
    return [
        Path("results--ver-1959_1.csv"),
        Path("results--ver-1959_2.csv"),
        Path("results--ver-1959_3.csv"),
    ]


def compare_results(gt_dir: Path, test_dir: Path, csv_names: list[str]) -> None:
    for csv_name in csv_names:
        gt_df = pd.read_csv(gt_dir / csv_name)
        test_df = pd.read_csv(test_dir / csv_name)
        pd.testing.assert_frame_equal(gt_df, test_df, obj=csv_name)
        print(f"Identity test passed for {csv_name}")
        check_integrity(test_df)
        print(f"Integrity test passed for {csv_name}")


def check_integrity(test_df: pd.DataFrame) -> None:
    test_df["er_temp"] = test_df["expositions_rec"].map(lambda x: x.split(";")).map(lambda x: [int(xx) for xx in x])
    test_df["g_temp"] = test_df.apply(
        lambda row: compute_gain(row["exposed_nb"], row["seed_nb"], (row["exposed_nb"] + row["unexposed_nb"])),
        axis=1
    )
    test_df["a_temp"] = test_df.apply(
        lambda row: compute_area(row["er_temp"], row["seed_nb"], (row["exposed_nb"] + row["unexposed_nb"])),
        axis=1
    )
    assert test_df["seed_nb"].equals(test_df["seed_ids"].map(lambda x: x.split(";")).map(lambda x: len(x)))
    assert test_df["seed_nb"].equals(test_df["er_temp"].map(lambda x: x[0]))
    assert test_df["simulation_length"].equals(test_df["er_temp"].map(lambda x: len(x)))
    assert test_df["exposed_nb"].equals(test_df["er_temp"].map(lambda x: sum(x)))
    assert test_df["gain"].round(3).equals(test_df["g_temp"].round(3))
    assert test_df["area"].round(3).equals(test_df["a_temp"].round(3))


@pytest.mark.parametrize(
        "tcase_config, tcase_csv_names",
        [
            ("tcase_ranking_config", "tcase_ranking_csv_names"),
        ]
)
def test_e2e(tcase_config, tcase_csv_names, request, tmpdir):
    config = request.getfixturevalue(tcase_config)
    csv_names = request.getfixturevalue(tcase_csv_names)
    config["io"]["out_dir"] = str(tmpdir)
    set_rng_seed(config["run"]["rng_seed"])
    simulate.run_experiments(config)
    compare_results(Path("data/test"), Path(tmpdir), csv_names)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pytest.main(["-vs", __file__])
