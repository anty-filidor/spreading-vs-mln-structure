from copy import deepcopy
from typing import Literal

import network_diffusion as nd
import networkx as nx

from scipy.stats import kendalltau
from sklearn.metrics import adjusted_mutual_info_score


def align_layers(
    net: nd.MultilayerNetwork,
    l1_name: str,
    l2_name: str,
    method: Literal["destructive", "additive"],
) -> dict[str, nx.Graph]:
    """
    Align set of nodes in the given two layers of a multilayer network.

    :param net: _description_
    :param l1_name: _description_
    :param l2_name: _description_
    :param method: there are two options:
        `additive` - target set of nodes is union of nodes in two layers
        `destructive` - target set of nodes is intersection of nodes in two layers
    :return: _description_
    """
    l1 = deepcopy(net[l1_name])
    l1_nodes = set(l1.nodes)

    l2 = deepcopy(net[l2_name])
    l2_nodes = set(l2.nodes)

    if method == "additive":
        correct_nodes = l1_nodes.union(l2_nodes)
    elif method == "destructive":
        correct_nodes = l1_nodes.intersection(l2_nodes)
    else:
        raise ValueError("Unknown alignment method!")

    l1.remove_nodes_from(l1_nodes.difference(correct_nodes))
    l1.add_nodes_from(correct_nodes.difference(l1_nodes))

    l2.remove_nodes_from(l2_nodes.difference(correct_nodes))
    l2.add_nodes_from(correct_nodes.difference(l2_nodes))

    return {l1_name: l1, l2_name: l2}


def degrees_correlation(graph_1: nx.Graph, graph_2: nx.Graph, alpha: float = 0.05) -> float:
    _l1_deg = nx.degree_histogram(graph_1)
    _l2_deg = nx.degree_histogram(graph_2)
    l1_deg, l2_deg = [], []
    for idx in range(max(len(_l1_deg), len(_l2_deg))):
        l1_deg.append(_l1_deg[idx] if idx + 1 <= len(_l1_deg) else 0)
        l2_deg.append(_l2_deg[idx] if idx + 1 <= len(_l2_deg) else 0)
    statistic, pvalue = kendalltau(x=l1_deg, y=l2_deg, nan_policy="raise", variant="b")
    if pvalue < alpha:
        return statistic
    return 0.0


def partitions_correlation(graph_1: nx.Graph, graph_2: nx.Graph) -> float:

    # obtain partitions in each graph
    graph_1_partitions = nx.community.louvain_communities(graph_1)
    graph_2_partitions = nx.community.louvain_communities(graph_2)

    # create dict keyed by nodes' ids, valued by array with partitions they're assigned into
    nodes_partitions = {node: [] for node in graph_1.nodes}
    for community_label, community_set in enumerate(graph_1_partitions):
        for node in community_set:
            nodes_partitions[node].append(community_label)
    for community_label, community_set in enumerate(graph_2_partitions):
        for node in community_set:
            nodes_partitions[node].append(community_label)

    # convert into two tables of indices accepted by sklearn
    partition_1_idcs, partition_2_idcs = [], []
    for node, (partition_1_idx, partition_2_idx) in nodes_partitions.items():
        partition_1_idcs.append(partition_1_idx)
        partition_2_idcs.append(partition_2_idx)

    # compute AMI and return it
    return float(adjusted_mutual_info_score(partition_1_idcs, partition_2_idcs))


def edges_jaccard(graph_1: nx.Graph, graph_2: nx.Graph) -> float:
    graph_1_edges = set(graph_1.edges)
    graph_2_edges = set(graph_2.edges)
    if len(graph_1_edges) == 0 or len(graph_2_edges) == 0:
        return None
    return len(graph_1_edges.intersection(graph_2_edges)) / len(graph_1_edges.union(graph_2_edges))


def edges_r(graph_1: nx.Graph, graph_2: nx.Graph) -> float:
    graph_1_edges = set(graph_1.edges)
    graph_2_edges = set(graph_2.edges)
    if min(len(graph_1_edges), len(graph_2_edges)) == 0:
           return None
    return len(graph_1_edges.intersection(graph_2_edges)) / min(len(graph_1_edges), len(graph_2_edges))
