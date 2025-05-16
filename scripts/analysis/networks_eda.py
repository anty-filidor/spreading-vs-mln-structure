"""Plot basic properties of the networks."""

# TODO: needs to be updated - network loader has been changed

from dataclasses import dataclass, asdict

import networkx as nx
import network_diffusion as nd
import numpy as np
import pandas as pd

from src.params_handler import Network, load_networks


BASE_NETWORK = "timik1q2009"
TWIN_NETWORKS = {
    "series_0": "twin", 
    "series_1": "75_prct_actors", 
    "series_2": "50_prct_actors", 
    "series_3": "25_prct_actors", 
    "series_4": "01_prct_actors", 
    "series_5": "xi_1.00", 
    "series_6": "xi_200_prct", 
    "series_7": "xi_50_prct", 
    "series_8": "xi_0.01",
    "series_9": "limited_degree",
}
# BINS = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]


@dataclass
class NetworkMetrics:
    actors: int
    layers: int
    nodes: float
    edges: float
    degree: float
    density: float
    components: float
    # betweenness: float
    # closeness: float
    clustering_coeff: float


def analyse_network(network: nd.MultilayerNetwork) -> NetworkMetrics:
    _nds, _edgs, _degree, _dnst, _comp, _btwn, _clsns, _clstr = [], [], [], [], [], [], [], []
    for l_graph in network.layers.values():
        _nds.append(len(l_graph.nodes()))
        _edgs.append(len(l_graph.edges()))
        _degree.extend(np.array(nx.degree(l_graph))[:, 1].astype(int))
        _dnst.append(nx.density(l_graph))
        _comp.append(len([comp for comp in nx.connected_components(l_graph)]))
        # _btwn.extend(list(nx.betweenness_centrality(l_graph).values()))
        # _clsns.extend(list(nx.closeness_centrality(l_graph).values()))
        _clstr.extend(list(nx.clustering(l_graph).values()))

    return NetworkMetrics(
        actors=network.get_actors_num(),
        layers=len(network.layers),
        nodes=np.array(_nds).mean(),
        edges=np.array(_edgs).mean(),
        degree=np.array(_degree).mean(),
        density=np.array(_dnst).mean(),
        components=np.array(_comp).mean(),
        # betweenness=np.array(_btwn).mean(),
        # closeness=np.array(_clsns).mean(),
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
        # betweenness=np.mean([s.betweenness for s in stats]),
        # closeness=np.mean([s.closeness for s in stats]),
        clustering_coeff=np.mean([s.clustering_coeff for s in stats]),
    )


series_stats = {
    BASE_NETWORK: analyse_network(load_networks([BASE_NETWORK])[0].n_graph_nx)
}
for s_name, s_nick in TWIN_NETWORKS.items():
    print(f"\n\nProcessing {s_name}")
    s_nets = load_networks([f"mlnabcd^data/nets_generated/{s_name}/*"])
    series_stats[s_nick] = analyse_networks(s_nets)

stats_df = pd.DataFrame({key: asdict(val) for key, val in series_stats.items()})
stats_df.to_csv("data/nets_properties/nets_generated_properties.csv")
