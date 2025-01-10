"""Estimate parameters of the configuration model."""

from pathlib import Path

import network_diffusion as nd
import pandas as pd

from src.multi_abcd import configuration_model
from src.loaders.net_loader import load_network


def find_configuration_setup(net: nd.MultilayerNetwork) -> pd.DataFrame:
    print("\tFinding configuration setup")
    q = {l_name: configuration_model.get_q(l_graph, net.get_actors_num()) for l_name, l_graph in net.layers.items()}
    degrees = {l_name: configuration_model.get_degrees_stats(l_graph) for l_name, l_graph in net.layers.items()}
    partitions = {l_name: configuration_model.get_partitions_stats(l_graph) for l_name, l_graph in net.layers.items()}
    merged_stats = {l_name: {**degrees[l_name], **partitions[l_name], **{"q": q[l_name]}} for l_name in net.layers}
    return pd.DataFrame(merged_stats)


if __name__ == "__main__":

    networks = [
        "arxiv_netscience_coauthorship",
        "aucs",
        "cannes",
        "ckm_physicians",
        "eu_transportation",
        "l2_course_net_1",
        "lazega",
        "timik1q2009",
        "toy_network",
    ]

    workdir = Path(__file__).parent.parent.parent / "data/multi_abcd/configurations"
    workdir.mkdir(exist_ok=True, parents=True)

    for net_name in sorted(networks):

        print(net_name)
        net = load_network(net_name, as_tensor=False)
        if net.is_directed(): raise ValueError("Only undirected networks can be processed!")

        configuration_df = find_configuration_setup(net)
        configuration_df.to_csv(workdir / f"{net_name}_configuration.csv")
