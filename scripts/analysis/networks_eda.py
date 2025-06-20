"""Plot basic properties of the networks."""

from dataclasses import dataclass, asdict

import networkx as nx
import network_diffusion as nd
import numpy as np
import pandas as pd

from src.loaders.constants import SEPARATOR
from src.params_handler import Network, load_networks


BASE_NETWORK = ["bigreal", "timik1q2009"]
TWIN_NETWORKS = {
    "series_1": "timik_twin",
    "series_2": "xi_1.00",
    "series_3": "xi_200_prct",
    "series_4": "xi_50_prct",
    "series_5": "xi_0.01",
    "series_6": "150_prct_actors",
    "series_7": "125_prct_actors",
    "series_8": "75_prct_actors",
    "series_9": "50_prct_actors",
    "series_10": "r_1.000",
    "series_11": "r_0.667",
    "series_12": "r_0.333",
    "series_13": "r_0.001",
}
DEVICE = "cpu"


@dataclass
class NetworkMetrics:
    actors: int
    layers: int
    nodes: float
    edges: float
    degree: float
    density: float
    components: float
    clustering_coeff: float


def analyse_network(network: nd.MultilayerNetwork) -> NetworkMetrics:
    _nds, _edgs, _degree, _dnst, _comp,  _clstr = [], [], [], [], [], []
    for l_graph in network.layers.values():
        _nds.append(len(l_graph.nodes()))
        _edgs.append(len(l_graph.edges()))
        _degree.extend(np.array(nx.degree(l_graph))[:, 1].astype(int))
        _dnst.append(nx.density(l_graph))
        _comp.append(len([comp for comp in nx.connected_components(l_graph)]))
        _clstr.extend(list(nx.clustering(l_graph).values()))

    return NetworkMetrics(
        actors=network.get_actors_num(),
        layers=len(network.layers),
        nodes=np.array(_nds).mean(),
        edges=np.array(_edgs).mean(),
        degree=np.array(_degree).mean(),
        density=np.array(_dnst).mean(),
        components=np.array(_comp).mean(),
        clustering_coeff=np.array(_clstr).mean(),
    )


def analyse_networks(networks: list[Network]) -> NetworkMetrics:
    stats: list[NetworkMetrics] = []
    for network in networks:
        print(f"\tanalysing network {network.n_name}")
        stats.append(analyse_network(network.n_graph_nx))
    return NetworkMetrics(
        actors=np.mean([s.actors for s in stats]),
        layers=np.mean([s.layers for s in stats]),
        nodes=np.mean([s.nodes for s in stats]),
        edges=np.mean([s.edges for s in stats]),
        degree=np.mean([s.degree for s in stats]),
        density=np.mean([s.density for s in stats]),
        components=np.mean([s.components for s in stats]),
        clustering_coeff=np.mean([s.clustering_coeff for s in stats]),
    )


series_stats = {
    BASE_NETWORK[1]: analyse_network(
        load_networks([SEPARATOR.join(BASE_NETWORK)], DEVICE)[0].n_graph_nx
    )
}
for s_name, s_nick in TWIN_NETWORKS.items():
    print(f"\n\nProcessing {s_name}")
    s_nets = load_networks([f"mlnabcd^{s_name}/*"], DEVICE)
    series_stats[s_nick] = analyse_networks(s_nets)

stats_df = pd.DataFrame({key: asdict(val) for key, val in series_stats.items()})
stats_df.to_csv("data/nets_properties/nets_generated_properties.csv")
