"""A script with functions to facilitate liading simulation's parameters and input data."""

import itertools
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import network_diffusion as nd

from src.loaders.net_loader import load_network
from src.loaders.constants import SEPARATOR


class JSONEncoder(json.JSONEncoder):
    def default(self, obj) -> dict[str, Any]:
        if isinstance(obj, nd.MLNetworkActor):
            return obj.__dict__
        return super().default(obj)


@dataclass(frozen=True)
class Network:
    n_type: str
    n_name: str
    n_graph_pt: nd.MultilayerNetworkTorch
    n_graph_nx: nd.MultilayerNetwork

    @property
    def rich_name(self) -> str:
        _type = self.n_type.replace("/", ".")
        _name = self.n_name.replace("/", ".")
        if _type == _name:
            return _type
        return f"{_type}{SEPARATOR}{_name}"


@dataclass(frozen=True)
class SeedSelector:
    name: str
    selector: nd.seeding.BaseSeedSelector


class MyRandomSeedSelector(nd.seeding.RandomSeedSelector):  # TODO: move to nd
    """Base version just stopped being capable to make deterministic; here's a workaround. """

    def actorwise(self, net: nd.MultilayerNetwork) -> list[nd.MLNetworkActor]:
        actors = net.get_actors(shuffle=False)
        sorted_actors = sorted(actors, key=lambda x: x.actor_id)
        random.shuffle(sorted_actors)
        return sorted_actors


def get_parameter_space(
    protocols: list[str],
    probabs: list[float],
    seed_budgets: list[float],
    ss_methods: list[str],
    networks: list[tuple[str, str]],
) -> list[tuple[str, tuple[float, float], float, tuple[str, str], str]]:
    seed_budgets_full = [(100 - i, i) for i in seed_budgets]
    p_space = itertools.product(protocols, seed_budgets_full, probabs, networks, ss_methods)
    return list(p_space)


def create_out_dir(out_dir: str) -> Path:
    try:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(exist_ok=True, parents=True)
    except FileExistsError:
        print("Redirecting output to hell...")
        out_dir_path = Path(tempfile.mkdtemp())
    return out_dir_path


def get_seed_selector(selector_name: str) -> nd.seeding.BaseSeedSelector:
    if selector_name == "btw":
        return nd.seeding.BetweennessSelector()
    if selector_name == "cbim":
        return nd.seeding.CBIMSeedselector(merging_idx_threshold=1)
    elif selector_name == "cim":
        return nd.seeding.CIMSeedSelector()
    elif selector_name == "cls":
        return nd.seeding.ClosenessSelector()
    elif selector_name == "deg_c":
        return nd.seeding.DegreeCentralitySelector()
    elif selector_name == "deg_cd":
        return nd.seeding.DegreeCentralityDiscountSelector()
    elif selector_name == "k_sh":
        return nd.seeding.KShellSeedSelector()
    elif selector_name == "k_sh_m":
        return nd.seeding.KShellMLNSeedSelector()
    elif selector_name == "kpp_sh":
        return nd.seeding.KPPShellSeedSelector()
    elif selector_name == "nghb_1s":
        return nd.seeding.NeighbourhoodSizeSelector(connection_hop=1)
    elif selector_name == "nghb_2s":
        return nd.seeding.NeighbourhoodSizeSelector(connection_hop=2)
    elif selector_name == "nghb_sd":
        return nd.seeding.NeighbourhoodSizeDiscountSelector()
    elif selector_name == "p_rnk":
        return nd.seeding.PageRankSeedSelector()
    elif selector_name == "p_rnk_m":
        return nd.seeding.PageRankMLNSeedSelector()
    elif selector_name == "random":
        return MyRandomSeedSelector()
    elif selector_name == "v_rnk":
        return nd.seeding.VoteRankSeedSelector()
    elif selector_name == "v_rnk_m":
        return nd.seeding.VoteRankMLNSeedSelector()
    raise AttributeError(f"{selector_name} is not a valid name for seed selector!")


def load_networks(networks: list[str], device: str) -> list[Network]:
    nets = []
    for net_regex in networks:
        net_type, net_name = net_regex.split(SEPARATOR)
        print(f"Loading network(s): {net_type} - {net_name}")
        for (net_type, net_name), net_graph in load_network(net_type=net_type, net_name=net_name).items():
            print("\tconverting to PyTorch")
            nets.append(
                Network(
                    n_type=net_type,
                    n_name=net_name,
                    n_graph_nx=net_graph,
                    n_graph_pt=nd.MultilayerNetworkTorch.from_mln(net_graph, device)
                )
            )
    print(f"Loaded {len(nets)} networks")
    return nets


def load_seed_selectors(ss_methods: list[str]) -> list[SeedSelector]:
    ssms = []
    for ssm_name in ss_methods:
        print(f"Initialising seed selection method: {ssm_name}")
        ssms.append(SeedSelector(ssm_name, get_seed_selector(ssm_name)))
    return ssms


def compute_rankings(
    seed_selectors: list[SeedSelector],
    networks: list[Network],
    out_dir: Path,
    version: str,
    ranking_path: Path | None = None,
) -> dict[tuple[str, str], list[nd.MLNetworkActor]]:
    """For given networks and seed seleciton methods compute or load rankings of actors."""
    
    nets_and_ranks = {}  # {(net_name, ss_name): ranking}
    for n_idx, net in enumerate(networks):
        print(f"Computing ranking for: {net.rich_name} ({n_idx+1}/{len(networks)})")

        for s_idx, ssm in enumerate(seed_selectors):
            print(f"Using method: {ssm.name} ({s_idx+1}/{len(seed_selectors)})")   
            ss_ranking_name = Path(f"ss-{ssm.name}--net-{net.rich_name}--ver-{version}.json")

            # obtain ranking for given ssm and net
            ranking = []
            if ranking_path:
                ranking_file = Path(ranking_path) / ss_ranking_name
                try:
                    with open(ranking_file, "r") as f:
                        ranking_dict = json.load(f)
                    ranking = [nd.MLNetworkActor.from_dict(rd) for rd in ranking_dict]
                    print("\tranking loaded")
                except:
                    print("\tunable to load ranking, falling back to computations")
            if len(ranking) == 0:
                ranking = ssm.selector(net.n_graph_nx, actorwise=True)
                print("\tranking computed")
            assert len(ranking) == net.n_graph_nx.get_actors_num()
            nets_and_ranks[(net.rich_name, ssm.name)] = ranking

            # save computed ranking
            with open(out_dir / ss_ranking_name, "w") as f:
                json.dump(ranking, f, cls=JSONEncoder)
                print(f"\tranking saved in the storage")

    return nets_and_ranks
