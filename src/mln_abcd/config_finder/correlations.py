from typing import Any

import networkx as nx
import pandas as pd

from scipy.stats import kendalltau
from sklearn.metrics import adjusted_mutual_info_score


def degree_crosslayer_correlation(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    alpha: float | None = 0.05,
) -> float:
    _l1_deg = dict(graph_1.degree())
    _l2_deg = dict(graph_2.degree())
    df_deg = pd.DataFrame({"graph_1": _l1_deg, "graph_2": _l2_deg}).sort_index()
    assert None not in df_deg  # a sanity check
    l1_deg, l2_deg = df_deg["graph_1"].to_list(), df_deg["graph_2"].to_list()
    statistic, pvalue = kendalltau(x=l1_deg, y=l2_deg, nan_policy="raise", variant="b")
    if not alpha:
        return statistic
    elif pvalue < alpha:
        return statistic
    return 0.0


def partitions_correlation(
    graph_1: nx.Graph, 
    graph_2: nx.Graph,
    graph_1_partitions: list[set[Any]] | None = None,
    graph_2_partitions: list[set[Any]] | None = None,
    seed: int | None = 42,
) -> float:
    # obtain partitions in each graph
    if not graph_1_partitions:
        graph_1_partitions = nx.community.louvain_communities(graph_1, seed=seed)
    if not graph_2_partitions:
        graph_2_partitions = nx.community.louvain_communities(graph_2, seed=seed)

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


def edges_r(graph_1: nx.Graph, graph_2: nx.Graph) -> float:
    g1_edges = set(graph_1.edges)
    g2_edges = set(graph_2.edges)
    if min(len(g1_edges), len(g2_edges)) == 0:
           return None
    return len(g1_edges.intersection(g2_edges)) / min(len(g1_edges), len(g2_edges))
