"""Functions to infer configuration parameters of the existing network."""

import juliacall  # this is added to silent a warning raised by importing both torch an juliacall
import networkx as nx
import network_diffusion as nd
import pandas as pd


from src.generator.mln_abcd import MLNABCDGraphGenerator, MLNConfig
from src.multi_abcd import configuration_model, correlations, helpers
from src.loaders.net_loader import load_network
from src.utils import set_rng_seed


def get_edges_cor(net: nd.MultilayerNetwork) -> pd.DataFrame:
    """Get correlation matrix for edges."""
    edges_cor_raw = []
    for la_name, lb_name in helpers.prepare_layer_pairs(net.layers.keys()):
        aligned_layers = helpers.align_layers(net, la_name, lb_name, "destructive")
        edges_stat = correlations.edges_r(aligned_layers[la_name], aligned_layers[lb_name])
        edges_cor_raw.append({(la_name, lb_name): edges_stat})
    edges_cor_df = helpers.create_correlation_matrix(edges_cor_raw)
    return edges_cor_df.round(3).replace(0.0, 0.001)


def get_layer_params(net: nd.MultilayerNetwork) -> pd.DataFrame:
    """Infer layers' parameters used by MLNABCD for a given network."""
    q, gamma_delta_Delta, beta_s_S_xi = {}, {}, {}
    tau = configuration_model.get_tau(net, alpha=None)
    r = configuration_model.get_r(net, seed=RNG_SEED)

    nb_actors = net.get_actors_num()
    for l_name, l_graph in net.layers.items():
        q[l_name] = configuration_model.get_q(l_graph, nb_actors)
        gamma_delta_Delta[l_name] = configuration_model.get_gamma_delta_Delta(l_graph)
        beta_s_S_xi[l_name] = configuration_model.get_beta_s_S_xi(l_graph)

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
    layer_params = get_layer_params(net=ref_net)
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
