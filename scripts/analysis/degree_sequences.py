"""Generate degree sequences for network we used in the paper."""

from pathlib import Path

from src.loaders.net_loader import load_network
from src.mln_abcd.config_finder.helpers import get_degree_sequence


networks = [
    "arxiv_netscience_coauthorship",
    "aucs",
    "cannes",
    "ckm_physicians",
    "eu_transportation",
    "l2_course_net_1",
    "lazega",
    "timik1q2009",
    "toy_network",
]


if __name__ == "__main__":
        
    workdir = Path(__file__).parent.parent.parent / "data/nets_properties/degree_sequences"
    workdir.mkdir(exist_ok=True, parents=True)

    for net_name in sorted(networks):

        print(net_name)
        net = load_network(net_name, as_tensor=False)

        degrees_table = get_degree_sequence(net)
        degrees_table.to_csv(workdir / f"{net_name}_degrees.csv")
