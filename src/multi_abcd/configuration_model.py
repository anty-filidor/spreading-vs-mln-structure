from typing import Any

import networkx as nx
import network_diffusion as nd
import numpy as np
import powerlaw
from scipy.stats import kendalltau

from src.multi_abcd import correlations
from src.multi_abcd.helpers import get_degree_sequence, prepare_layer_pairs


def get_q(net: nx.Graph, num_actors: int) -> float:
    """Get fraction of active nodes."""
    return len(net.nodes) / num_actors


def get_tau(net: nd.MultilayerNetwork, alpha: float | None = 0.05) -> dict[str, float]:
    """
    Get correlations between node labels and their degrees.

    Note, that due to multilaterance of the network, the routine first converts labels of nodes
    from the first layer so that the correlation is maximal (a node with the maximal degree gets the
    highest ID) and then it applies these labels to compute correlations in remaining layers. It 
    also computes correlations only for nodes with positive degree.  
    """
    net = net.to_multiplex()[0]
    layer_names = sorted(list(net.layers))

    degree_sequence  = get_degree_sequence(net).T
    degree_sequence = degree_sequence.sort_index().sort_values(by=layer_names[0], ascending=False)
    actors_map = {id: idx for idx, id in enumerate(list(degree_sequence.index)[::-1])}
    degree_sequence = degree_sequence.rename(index=actors_map)

    tau = {}
    for l_name in layer_names:
        l_ds = degree_sequence[l_name][degree_sequence[l_name] > 0]
        statistic, pvalue = kendalltau(
            x=l_ds.index.to_list(),
            y=l_ds.to_list(),
            nan_policy="raise",
            variant="b",
        )
        if not alpha:
            tau[l_name] = statistic.item()
        elif pvalue < alpha:
            tau[l_name] = statistic.item()
        else:
            tau[l_name] = 0.0
    
    return tau


def get_r(net: nd.MultilayerNetwork, seed: int | None = None) -> dict[str, float]:
    """
    Get correlations between partitions.
    
    Nota, that due to impossibility to reverse the process of creating partitions by MLNABCD, this
    function only approximates the correlations by treating the first (alphabetically) layer of the
    network as the reference one.   
    """
    net = net.to_multiplex()[0]
    layer_names = sorted(list(net.layers))

    ref_layer = net[layer_names[0]]
    ref_partitions = nx.community.louvain_communities(ref_layer, seed=seed)

    r = {}
    for l_name in layer_names:
        ami = correlations.partitions_correlation(
            graph_1=ref_layer,
            graph_2=net[l_name],
            graph_1_partitions=ref_partitions,
            seed=seed,
        )
        r[l_name] = ami
    
    return r


def _fit_exponent_powerlaw(raw_data: list[float | int]) -> float:
    results = powerlaw.Fit(raw_data, discrete=True, verbose=False)
    return results.alpha


def get_gamma_delta_Delta(net: nx.Graph) -> dict[str, float]:
    """Get powerlaw exponent and min/max degree for a given layer."""
    degrees = [d for _, d in net.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    return {"gamma": _fit_exponent_powerlaw(degrees), "delta": min_degree, "Delta": max_degree}


def _partitions_noise(net: nx.Graph, partition_a: set[Any], partition_b: set[Any]) -> float:
    sub_net = net.subgraph(partition_a.union(partition_b))
    edges_nb = len(sub_net.edges)
    common_edges = 0
    for edge in sub_net.edges:
        node_a, node_b = edge
        if node_a in partition_a and node_b in partition_b:
            common_edges += 1
        elif node_a in partition_b and node_b in partition_a:
            common_edges += 1
    return common_edges / edges_nb


def _avg_partitions_noise(net: nx.Graph, partitions: list[set[Any]]) -> float:
    """
    The noise is avg fract. of edges between two partitions to number of edges in the subgrah
    incuded by these partitions.
    """
    all_noises = []
    partitions_dict = {f"p_{idx}": partition for idx, partition in enumerate(partitions)}
    for partition_a_name, partition_b_name in prepare_layer_pairs(partitions_dict.keys()):
        partition_a = partitions_dict[partition_a_name]
        partition_b = partitions_dict[partition_b_name]
        ab_noise = _partitions_noise(net, partition_a, partition_b)
        all_noises.append(ab_noise)
    return np.array(all_noises).mean().item()


def get_beta_s_S_xi(net: nx.Graph) -> dict[str, float]:
    """Get powerlaw exponent and min/max community size for a given layer."""
    partitions = nx.community.louvain_communities(net)
    partitions_sizes = [len(part) for part in partitions]
    return {
        "beta": _fit_exponent_powerlaw(partitions_sizes),
        "s": min(partitions_sizes),
        "S": max(partitions_sizes),
        "xi": _avg_partitions_noise(net, partitions)
    }
