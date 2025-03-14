"""Functions to infer configuration parameters of the existing network."""

import warnings
from typing import Any

import juliacall  # this is added to silent a warning raised by importing both torch an juliacall
import networkx as nx
import network_diffusion as nd
import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import kendalltau

from src.mln_abcd.config_finder import correlations, helpers


def get_q(net: nx.Graph, num_actors: int) -> float:
    """Get fraction of active nodes."""
    return len(net.nodes) / num_actors


def get_tau(net: nd.MultilayerNetwork, alpha: float | None = 0.05) -> dict[str, float]:
    """
    Get correlations between node labels and their degrees.

    Note, that due to multilaterance of the network, the routine first converts labels of nodes
    from the first (alphabetically) layer, in a way to maximise the correlation (a node with the
    maximal degree has assigned the highest ID). Then it uses these labels in computations of the
    correlations in remaining layers. In addition it considers only nodes with a positive degree.  
    """
    net = net.to_multiplex()[0]
    layer_names = sorted(list(net.layers))

    degree_sequence  = helpers.get_degree_sequence(net).T
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
    
    Note, that due to impossibility to reverse the process of creating partitions by MLNABCD, this
    function only approximates the correlations as follows. It takes the first (alphabetically)
    layer of the network as a reference. Then it uses it to compute correlations with other layers.
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
    for partition_a_name, partition_b_name in helpers.prepare_layer_pairs(partitions_dict.keys()):
        partition_a = partitions_dict[partition_a_name]
        partition_b = partitions_dict[partition_b_name]
        ab_noise = _partitions_noise(net, partition_a, partition_b)
        all_noises.append(ab_noise)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
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


def get_edges_cor(net: nd.MultilayerNetwork) -> pd.DataFrame:
    """Get correlation matrix for edges."""
    edges_cor_raw = []
    for la_name, lb_name in helpers.prepare_layer_pairs(net.layers.keys()):
        aligned_layers = helpers.align_layers(net, la_name, lb_name, "destructive")
        edges_stat = correlations.edges_r(aligned_layers[la_name], aligned_layers[lb_name])
        edges_cor_raw.append({(la_name, lb_name): edges_stat})
    edges_cor_df = helpers.create_correlation_matrix(edges_cor_raw)
    return edges_cor_df.round(3).fillna(0.0)


def get_layer_params(net: nd.MultilayerNetwork, seed: int | None = None) -> pd.DataFrame:
    """Infer layers' parameters used by MLNABCD for a given network."""
    q, gamma_delta_Delta, beta_s_S_xi = {}, {}, {}
    tau = get_tau(net, alpha=None)
    r = get_r(net, seed=seed)

    nb_actors = net.get_actors_num()
    for l_name, l_graph in net.layers.items():
        q[l_name] = get_q(l_graph, nb_actors)
        gamma_delta_Delta[l_name] = get_gamma_delta_Delta(l_graph)
        beta_s_S_xi[l_name] = get_beta_s_S_xi(l_graph)

    params_dict = {
        l_name: {
            **{"q": q[l_name]},
            **{"tau": tau[l_name]},
            **{"r": r[l_name]},
            **gamma_delta_Delta[l_name],
            **beta_s_S_xi[l_name],
        }
        for l_name in net.layers
    }
    params_df = pd.DataFrame(params_dict).T.sort_index()
    return params_df.round(3).replace(0.0, 0.001)


if __name__ == "__main__":

    from src.mln_abcd.julia_wrapper import MLNABCDGraphGenerator, MLNConfig
    from src.loaders.net_loader import load_network
    from src.utils import set_rng_seed

    RNG_SEED = 42
    NET_NAME = "aucs"

    set_rng_seed(seed=RNG_SEED)
    ref_net = load_network(NET_NAME, as_tensor=False)

    mapping_dict = {l_name: l_idx for l_idx, l_name in enumerate(sorted(ref_net.layers), 1)}

    # infer edges' correlation matrix
    edges_cor = get_edges_cor(net=ref_net)
    print(edges_cor)
    edges_cor = edges_cor.rename(mapping_dict, axis=0)
    edges_cor = edges_cor.rename(mapping_dict, axis=1)
    edges_cor.to_csv("ref_edges_cor.csv")

    # infer layers' parameters
    layer_params = get_layer_params(net=ref_net, seed=RNG_SEED)
    print(layer_params)
    layer_params = layer_params.rename(mapping_dict, axis=0)
    layer_params.to_csv("ref_layer_params.csv", index=False)

    # load configuration
    mln_config = MLNConfig(
        seed=RNG_SEED,
        n=ref_net.get_actors_num(),
        edges_cor="ref_edges_cor.csv",
        layer_params="ref_layer_params.csv",
        d_max_iter=1000,
        c_max_iter=1000,
        t=100,
        eps=0.01,
        d=2,
        edges_filename="./edges.dat",
        communities_filename="./communities.dat",
    )

    # generate a network
    MLNABCDGraphGenerator()(config=mln_config)
