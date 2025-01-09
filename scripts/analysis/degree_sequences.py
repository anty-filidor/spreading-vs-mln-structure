"""Generate degree sequences for network we used in the paper."""

from pathlib import Path

import network_diffusion as nd
import pandas as pd

from src.loaders.net_loader import load_network


def get_degrees_table(net: nd.MultilayerNetwork) -> pd.DataFrame:
    net_degrees = {}
    for l_name, l_graph in net.layers.items():
        net_degrees[l_name] = dict(l_graph.degree())
    return pd.DataFrame(net_degrees).T


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


if __name__ == "__main__":
        
    workdir = Path(__file__).parent.parent.parent / "data/multi_abcd/degree_sequences"
    workdir.mkdir(exist_ok=True, parents=True)

    for net_name in sorted(networks):

        print(net_name)
        net = load_network(net_name, as_tensor=False)
        if net.is_directed(): raise ValueError("Only undirected networks can be processed!")

        degrees_table = get_degrees_table(net)
        degrees_table.to_csv(workdir / f"{net_name}_degrees.csv")
