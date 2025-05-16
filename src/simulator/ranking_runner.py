"""Pure ranking based step handler."""

import network_diffusion as nd

from src.params_handler import Network
from src.result_handler import SimulationFullResult
from src.simulator.simulation_step import experiment_step


def handle_step(
    proto: str, 
    p: float,
    budget: tuple[float, float],
    ss_method: str,
    net: Network,
    ranking: list[nd.MLNetworkActor],
    max_epochs_num: int,
) -> list[SimulationFullResult]:
    """The easiest way to handle case basing only on the ranking."""
    step_spr = experiment_step(
        protocol=proto,
        p=p,
        budget=budget,
        net=net,
        ranking=ranking,
        max_epochs_num=max_epochs_num,
    )
    step_sfr = SimulationFullResult.enhance_SPR(
        SPR=step_spr,
        network=net.rich_name,
        protocol=proto,
        probab=p,
        seed_budget=budget[1],
        ss_method=ss_method,
    )
    return [step_sfr]
