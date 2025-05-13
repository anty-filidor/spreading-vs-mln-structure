"""A loader for multilayer networks stored in the dataset."""

from functools import wraps
from glob import glob
from pathlib import Path
from typing import Callable

import pandas as pd
import network_diffusion as nd
import networkx as nx

from src.loaders.constants import (
    MLN_RAW_DATA_PATH,
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    ARXIV_NETSCIENCE_COAUTHORSHIP_MATH,
    AUCS,
    CANNES,
    CKM_PHYSICIANS,
    EU_TRANSPORTATION,
    EU_TRANSPORT_KLM,
    ER1,
    ER2,
    ER3,
    ER5,
    FMRI74,
    L2_COURSE_NET_1,
    L2_COURSE_NET_2,
    L2_COURSE_NET_3,
    LAZEGA,
    SF1,
    SF2,
    SF3,
    SF5,
    TIMIK1Q2009,
    TOY_NETWORK,
    MLNABCD_PREFIX,
    WILDCARD
)
from src.loaders.fmri74 import read_fmri74
from src.mln_abcd.julia_reader import load_edgelist


def _network_from_pandas(path):
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {l_name: nx.Graph() for l_name in df["layer"].unique()}
    for _, row in df.iterrows():  # TODO: consider changing the method of iterating
        net_dict[row["layer"]].add_edge(row["node_1"], row["node_2"])
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_dict.keys()), network_list=list(net_dict.values())
    )


def return_some_layers(get_network_func: Callable) -> Callable:
    """Decorator for network loader to filter out a multilayer network to return only one layer."""
    @wraps(get_network_func)
    def wrapper(layer_slice = None):
        net = get_network_func()
        if not layer_slice:
            return net
        l_graphs = [net.layers[layer] for layer in layer_slice]
        return nd.MultilayerNetwork.from_nx_layers(l_graphs, layer_slice)
    return wrapper


def get_aucs_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_real/aucs.mpx")


def get_ckm_physicians_network():
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/CKM-Physicians-Innovation_4NoNature.edges"
    )


@return_some_layers
def get_eu_transportation_network():
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/EUAirTransportation_multiplex_4NoNature.edges"
    )


def get_lazega_network():
    return _network_from_pandas(
        f"{MLN_RAW_DATA_PATH}/small_real/Lazega-Law-Firm_4NoNatureNoLoops.edges"
    )


def get_er2_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/er_2.mpx")


def get_er3_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/er_3.mpx")


@return_some_layers
def get_er5_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/er_5.mpx")


def get_sf2_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/sf_2.mpx")


def get_sf3_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/sf_3.mpx")


@return_some_layers
def get_sf5_network():
    return nd.MultilayerNetwork.from_mpx(file_path=f"{MLN_RAW_DATA_PATH}/small_artificial/sf_5.mpx")


def get_ddm_network(layernames_path, edgelist_path, weighted, digraph):
    # read mapping of layer IDs to their names
    with open(layernames_path, encoding="utf-8") as file:
        layer_names = file.readlines()
    layer_names = [ln.rstrip('\n').split(" ") for ln in layer_names]
    layer_names = {ln[0]: ln[1] for ln in layer_names}
    
    # read the edgelist and create containers for the layers
    df = pd.read_csv(
        edgelist_path,
        names=["layer_id", "node_1", "node_2", "weight"],
        sep=" "
    )
    net_ids_dict = {
        l_name: nx.DiGraph() if digraph else nx.Graph()
        for l_name in list(df["layer_id"].unique())
    }

    # populate network with edges
    for _, row in df.iterrows():  # TODO: consider changing the method of iterating
        if weighted:
            attrs = {"weight": row["weight"]}
        else:
            attrs = {}
        net_ids_dict[row["layer_id"]].add_edge(row["node_1"], row["node_2"], **attrs)
    
    # rename layers
    net_names_dict = {
        layer_names[str(layer_id)]: layer_graph
        for layer_id, layer_graph in net_ids_dict.items()
    }

    # create multilater network from edges
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_names_dict.keys()), network_list=list(net_names_dict.values())
    )


@return_some_layers
def get_arxiv_network():
    root_path = Path(f"{MLN_RAW_DATA_PATH}/arxiv_netscience_coauthorship/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "arxiv_netscience_layers.txt",
        edgelist_path=root_path / "arxiv_netscience_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_cannes_network():
    root_path = Path(f"{MLN_RAW_DATA_PATH}/cannes_2013_social/Dataset")
    net = get_ddm_network(
        layernames_path= root_path / "Cannes2013_layers.txt",
        edgelist_path=root_path / "Cannes2013_multiplex.edges",
        weighted=False,
        digraph=False,
    )
    return net


def get_timik1q2009_network():
    layer_graphs = []
    layer_names = []
    for i in Path(f"{MLN_RAW_DATA_PATH}/timik1q2009").glob("*.csv"):
        layer_names.append(i.stem)
        layer_graphs.append(nx.from_pandas_edgelist(pd.read_csv(i)))
    return nd.MultilayerNetwork.from_nx_layers(layer_graphs, layer_names)


def read_mlnabcd_networks(convolved_name: str) -> dict[tuple[str, str], nd.MultilayerNetwork]:
    net_paths_regex = convolved_name.split(WILDCARD)[-1]
    nets = {}
    for net_path in glob(net_paths_regex):
        net_path = Path(net_path)
        net_graph = load_edgelist(net_path)
        if net_graph.get_actors_num() == 0:
            print(f"Inferred {net_path} as a non-network file.")
            continue
        nets[(str(net_path.parent), net_path.stem)] = net_graph
    return nets


def _prepare_network(net: nd.MultilayerNetwork) -> nd.MultilayerNetwork:
    for _, l_graph in net.layers.items():
        l_graph.remove_edges_from(nx.selfloop_edges(l_graph))
        isolated_nodes = list(nx.isolates(l_graph))
        l_graph.remove_nodes_from(isolated_nodes)
    if net.is_directed(): raise ValueError("Only undirected networks can be processed right now!")
    return net


def prepare_network(load_networks_func: Callable) -> Callable:
    """Remove isolated nodes and nodes with self-edges only from the network."""
    @wraps(load_networks_func)
    def wrapper(*args, **kwargs) -> dict[str, nd.MultilayerNetwork]:
        net_dict = load_networks_func(*args, **kwargs)
        return {net_name: _prepare_network(net_graph) for net_name, net_graph in net_dict.items()}
    return wrapper


@prepare_network
def load_network(net_regex: str) -> dict[tuple[str, str], nd.MultilayerNetwork]:
    if net_regex.startswith(MLNABCD_PREFIX):
        return read_mlnabcd_networks(net_regex)
    elif net_regex == FMRI74:
        net_graph = read_fmri74(network_dir=f"{MLN_RAW_DATA_PATH}/CONTROL_fmt", binary=True, thresh=0.5)
    elif net_regex == ARXIV_NETSCIENCE_COAUTHORSHIP:
        net_graph = get_arxiv_network()
    elif net_regex == ARXIV_NETSCIENCE_COAUTHORSHIP_MATH:
        net_graph = get_arxiv_network(["math.OC"])
    elif net_regex == AUCS:
        net_graph = get_aucs_network()
    elif net_regex == CANNES:
        net_graph = get_cannes_network()
    elif net_regex == CKM_PHYSICIANS:
        net_graph = get_ckm_physicians_network()
    elif net_regex == EU_TRANSPORTATION:
        net_graph = get_eu_transportation_network()
    elif net_regex == EU_TRANSPORT_KLM:
        net_graph = get_eu_transportation_network(["KLM"])
    elif net_regex == L2_COURSE_NET_1:
        net_graph = nd.nets.get_l2_course_net(
            node_features=True, edge_features=True, directed=False
        ).snaps[0]
    elif net_regex == L2_COURSE_NET_2:
        net_graph = nd.nets.get_l2_course_net(
            node_features=True, edge_features=True, directed=False
        ).snaps[1]
    elif net_regex == L2_COURSE_NET_3:
        net_graph = nd.nets.get_l2_course_net(
            node_features=True, edge_features=True, directed=False
        ).snaps[2]
    elif net_regex == LAZEGA:
        net_graph = get_lazega_network()
    elif net_regex == ER1:
        net_graph = get_er5_network(["l2"])
    elif net_regex == ER2:
        net_graph = get_er2_network()
    elif net_regex == ER3:
        net_graph = get_er3_network()
    elif net_regex == ER5:
        net_graph = get_er5_network()
    elif net_regex == SF1:
        net_graph = get_sf5_network(["l3"])
    elif net_regex == SF2:
        net_graph = get_sf2_network()
    elif net_regex == SF3:
        net_graph = get_sf3_network()
    elif net_regex == SF5:
        net_graph = get_sf5_network()
    elif net_regex == TIMIK1Q2009:
        net_graph = get_timik1q2009_network()
    elif net_regex == TOY_NETWORK:
        net_graph = nd.nets.get_toy_network_piotr()
    else:
        raise AttributeError(f"Unknown network: {net_regex}")
    return {(net_regex, net_regex): net_graph}
