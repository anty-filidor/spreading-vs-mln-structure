"""A loader for multilayer networks stored in the dataset."""

from functools import wraps
from glob import glob
from pathlib import Path
from typing import Callable

import network_diffusion as nd
import networkx as nx
from tqdm import tqdm

from src.loaders.constants import MLN_ABCD_DATA_PATH
from src.loaders.small_artificial import load_small_artificial
from src.loaders.small_real import load_small_real
from src.loaders.big_real import load_big_real
from src.mln_abcd.julia_reader import load_edgelist


def read_mlnabcd_networks(net_name: str) -> dict[str, nd.MultilayerNetwork]:
    net_paths_regex = MLN_ABCD_DATA_PATH / net_name
    nets = {}
    progress_bar = tqdm(glob(str(net_paths_regex)))
    for net_path in progress_bar:
        net_path = Path(net_path)
        net_graph = load_edgelist(net_path)
        if net_graph.get_actors_num() == 0:
            progress_bar.set_description_str(f"{net_path} in a non-network file.")
            continue
        progress_bar.set_description_str("")
        nets[f"{net_path.parent.name}-{net_path.stem}"] = net_graph
    return nets


def _prepare_network(net: nd.MultilayerNetwork) -> nd.MultilayerNetwork:
    for _, l_graph in net.layers.items():
        l_graph.remove_edges_from(nx.selfloop_edges(l_graph))
        isolated_nodes = list(nx.isolates(l_graph))
        l_graph.remove_nodes_from(isolated_nodes)
    if net.is_directed(): raise ValueError("Only undirected networks can be processed right now!")
    return net


def prepare_network(load_network_func: Callable) -> Callable:
    """Remove isolated nodes and nodes with self-edges only from the network."""
    @wraps(load_network_func)
    def wrapper(*args, **kwargs) -> dict[tuple[str, str], nd.MultilayerNetwork]:
        net_dict = load_network_func(*args, **kwargs)
        print("\tremoving self-loops and isolated nodes")
        return {
            (net_type, net_name): _prepare_network(net_graph) for
            (net_type, net_name), net_graph in net_dict.items()
        }
    return wrapper


@prepare_network
def load_network(net_type: str, net_name: str) -> dict[tuple[str, str], nd.MultilayerNetwork]:
    if net_type == "mlnabcd":
        networks = read_mlnabcd_networks(net_name=net_name)
    elif net_type == "smallreal":
        networks = load_small_real(net_name=net_name)
    elif net_type == "smallart":
        networks = load_small_artificial(net_name=net_name)
    elif net_type == "bigreal":
        networks = load_big_real(net_name=net_name)
    else:
        raise AttributeError(f"Unknown network type: {net_type}")
    if len(networks) == 0:
        raise AttributeError(f"Loaded 0 networks!")    
    return {(net_type, net_name): net_graph for net_name, net_graph in networks.items()}
