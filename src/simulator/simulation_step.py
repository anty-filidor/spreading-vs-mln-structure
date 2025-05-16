"""A script with defined single simulation step."""

import warnings
from itertools import accumulate

import network_diffusion as nd
import numpy as np

from src.params_handler import Network
from src.result_handler import SimulationPartialResult
from src.simulator.torch_micm import TorchMICModel, TorchMICSimulator


def compute_gain(exposed_nb: int, seeds_nb: int, actors_nb: int) -> float:
    """Compute gain from simulation to reflect relative spreading coverage."""
    max_available_gain = actors_nb - seeds_nb  # TODO: move to nd
    obtained_gain = exposed_nb - seeds_nb
    return 100 * obtained_gain / max_available_gain


def compute_area(expositions_rec: list[int], seeds_nb: int, actors_nb: int) -> float | None:
    """Compute normalised AuC from expositions record while seed set impact is discarded."""
    cumsum = np.array(list(accumulate(expositions_rec)))  # TODO: move to nd
    if len(cumsum) < 2:
        warnings.warn("cumulated distribution must contain at least two samples.")
        return None
    cumsum_scaled = (cumsum - seeds_nb) / (actors_nb - seeds_nb)
    cumsum_steps = np.linspace(0, 1, len(cumsum_scaled))
    return np.trapezoid(cumsum_scaled, cumsum_steps)


def experiment_step(
    protocol: str,
    p: float,
    budget: tuple[float, float],
    net: Network,
    ranking: list[nd.MLNetworkActor],
    max_epochs_num: int,
) -> SimulationPartialResult:
    """
    Basic esperimental step to simulate spreading single time under MICM for given parameters.

    :param protocol: protocol function 
    :param p: activation probability
    :param budget: proportion of inactive to active actors at the beginning of simulation
    :param net: network to simulate spreading in
    :param ranking: ranking list to select seed set from

    :return: basic results from the experiment
    """

    # initialise spreading model and prepare data for simulation
    micm = TorchMICModel(protocol=protocol, probability=p)
    seed_set_size = int(len(ranking) * budget[1] / 100)
    seed_set = {actor.actor_id for actor in ranking[:seed_set_size]}
    net_pt = net.n_graph_pt
    optimal_steps = len(net_pt.actors_map) * len(net_pt.layers_order)

    # run experiment and obtain logs
    experiment = TorchMICSimulator(
        model=micm,
        net=net_pt,
        n_steps=optimal_steps if max_epochs_num < 0 else int(max_epochs_num),
        seed_set=seed_set,
        device=net_pt.device,
        debug=True,
    )
    logs = experiment.perform_propagation()
    gain = compute_gain(
        exposed_nb=logs["exposed"],
        seeds_nb=seed_set_size,
        actors_nb=len(net_pt.actors_map)
    )
    area = compute_area(
        expositions_rec=logs["expositions_rec"],
        seeds_nb=seed_set_size,
        actors_nb=len(net_pt.actors_map),
    )

    return SimulationPartialResult(
        seed_ids=";".join(sorted([str(s) for s in seed_set])),
        gain=gain,
        area=area,
        simulation_length=logs["simulation_length"],
        seed_nb=len(seed_set),
        exposed_nb=logs["exposed"],
        unexposed_nb=logs["not_exposed"],
        expositions_rec=";".join([str(r) for r in logs["expositions_rec"]]),
    )
