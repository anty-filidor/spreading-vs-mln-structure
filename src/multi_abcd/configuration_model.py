from typing import Any
import networkx as nx
import network_diffusion as nd
import numpy as np
import powerlaw
import scipy

from src.multi_abcd.helpers import prepare_layer_pairs


def get_nb_actors(net: nd.MultilayerNetwork) -> int:
    return net.get_actors_num()


def get_nb_layers(net: nd.MultilayerNetwork) -> int:
    return len(net.layers)


def get_q(net: nx.Graph, num_actors: int) -> float:
    """Get fraction of active nodes."""
    return len(net.nodes) / num_actors


def fit_exponent_regression(histogram: list[int]) -> float:
    degree_values = list(range(1, len(histogram) + 1))

    # convert to log space
    log_x = np.log(degree_values)
    log_y = np.log(histogram)

    # remove nans and regress Ax+B
    mask = ~np.isnan(log_y) & ~np.isinf(log_y)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_x[mask], log_y[mask])
    
    # return onlt significant trend
    if p_value < 0.05:
        return -1 * slope.item()

    return None


def fit_exponent_MLE(histogram: list[int], x_min: int = 5) -> float:
    """Max Likelyhood Estimator; xmin - cutoff for power-law behaviour."""
    filtered_sizes = []
    for x_idx, x_val in enumerate(histogram, 1):
        if x_idx >= x_min:
            for _ in range(x_val):
                filtered_sizes.append(x_idx)
    filtered_sizes = np.array(filtered_sizes)
    n = len(filtered_sizes)
    alpha = 1 + n / np.sum(np.log(filtered_sizes / x_min))
    return alpha.item()


def fit_exponent_powerlaw(raw_data: list[float | int]) -> float:
    results = powerlaw.Fit(raw_data, discrete=True, verbose=False)
    return results.alpha


def get_degrees_stats(net: nx.Graph) -> dict[str, float]:
    """Here I skip inactive actors in the layer."""
    degrees = [d for _, d in net.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    return {"gamma": fit_exponent_powerlaw(degrees), "delta": min_degree, "Delta": max_degree}


def avg_partitions_noise(net: nx.Graph, partitions: list[set[Any]]) -> float:
    """
    The noise is avg fract. of edges between two partitions to number of edges in the subgrah
    incuded by these partitions.
    """
    all_noises = []
    partitions_dict = {f"p_{idx}": partition for idx, partition in enumerate(partitions)}
    for partition_a_name, partition_b_name in prepare_layer_pairs(partitions_dict.keys()):
        partition_a = partitions_dict[partition_a_name]
        partition_b = partitions_dict[partition_b_name]
        ab_noise = partitions_noise(net, partition_a, partition_b)
        all_noises.append(ab_noise)
    return np.array(all_noises).mean().item()


def partitions_noise(net: nx.Graph, partition_a: set[Any], partition_b: set[Any]) -> float:
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


def get_partitions_stats(net: nx.Graph) -> dict[str, float]:
    partitions = nx.community.louvain_communities(net)
    partitions_sizes = [len(part) for part in partitions]
    return {
        "beta": fit_exponent_powerlaw(partitions_sizes),
        "s": min(partitions_sizes),
        "S": max(partitions_sizes),
        "xi": avg_partitions_noise(net, partitions)
    }
