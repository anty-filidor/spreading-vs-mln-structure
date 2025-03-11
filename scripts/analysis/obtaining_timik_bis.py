
import time

import network_diffusion as nd
# from network_diffusion.mln.mlnetwork_torch import _prepare_mln_for_conversion

from scipy.stats import kendalltau

from src.generator.mln_abcd import MLNABCDGraphGenerator, MLNConfig
from src.multi_abcd import configuration_model, correlations
from src.loaders.net_loader import load_network
from src.utils import set_rng_seed


set_rng_seed(seed=42)


net_name = "aucs"

t_start = time.time()
ref_net = load_network(net_name, as_tensor=False)
# ref_net = ref_net.to_multiplex()[0]
# ref_net, ac_map, _ = _prepare_mln_for_conversion(ref_net)
t_end = time.time()
print(f"Loaded in {t_end - t_start} seconds.")

ref_n = ref_net.get_actors_num()


def get_tau(net: nd.MultilayerNetwork, alpha: float | None = 0.05):
    """
    Get correlations between node labels and their degrees.

    Note, that due to multilaterance of the network, the routine first converts labels of nodes
    from the first layer so that the correlation is maximal (a node with the maximal degree gets the
    highest ID) and then it applies these labels to compute correlations in remaining layers. It 
    also computes correlations only for nodes with positive degree.  
    """
    net = net.to_multiplex()[0]
    layer_names = sorted(list(net.layers))

    degree_sequence  = correlations.degree_sequence(net).T
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


# r
"""
1. get partitions from a randomly chosen layer (reference)
2. create the latent layer with |A| points
3. match actors from the reference layer with points from the latent layer so that the structure of
   communities is preserved
4. for all other layers do:
    - find partitions
    - compute correlation between ???????
"""


# read parameters of each layer
q = {l_name: configuration_model.get_q(l_graph, ref_n) for l_name, l_graph in ref_net.layers.items()}
tau = get_tau(ref_net)
# r
degrees = {l_name: configuration_model.get_degrees_stats(l_graph) for l_name, l_graph in ref_net.layers.items()}
partitions = {l_name: configuration_model.get_partitions_stats(l_graph) for l_name, l_graph in ref_net.layers.items()}

# # convert to the dataframe
# merged_stats = {l_name: {**degrees[l_name], **partitions[l_name], **{"q": q[l_name]}} for l_name in ref_net.layers}


# # load from code
# mln_config = MLNConfig(
#     seed=42,
#     n=ref_n,  # 61702
#     edges_cor="data/multi_abcd/correlations/timik1q2009_edges.csv",
#     layer_params= "scripts/configs/example_generate/layer_params.csv",
#     d_max_iter=1000,
#     c_max_iter=1000,
#     t=100,
#     eps=0.01,
#     d=2,
#     edges_filename="./edges.dat",
#     communities_filename="./communities.dat",
# )

# # or from files
# mln_config = MLNConfig.from_yaml("scripts/configs/example_generate/mln_config.yaml")

# # then, generate a network
# MLNABCDGraphGenerator()(config=mln_config)