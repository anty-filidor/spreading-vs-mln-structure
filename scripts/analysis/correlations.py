"""Estimate parameters of the configuration model."""

from pathlib import Path

import matplotlib.pyplot as plt
import network_diffusion as nd
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.multi_abcd import configuration_model, correlations, helpers
from src.loaders.net_loader import load_network


NETS_MAPPING = {
    "arxiv_netscience_coauthorship": "arxiv",
    "aucs": "aucs",
    "cannes": "cannes",
    "ckm_physicians": "ckmp",
    "eu_transportation": "eutr-A",
    "l2_course_net_1": "l2-course",
    "lazega": "lazega",
    "timik1q2009": "timik",
    "toy_network": "toy_network",
}


def compute_statistics(net: nd.MultilayerNetwork, mode: str) -> dict[str, list[dict]]:
    
    print("\tComputing statistics")
    degree_stats, partition_stats, edges_stats = [], [], []

    for la_name, lb_name in helpers.prepare_layer_pairs(net.layers.keys()):
        print("\t\t", la_name, lb_name)
        aligned_layers = correlations.align_layers(net, la_name, lb_name, mode)

        degree_stat = correlations.degree_crosslayer_correlation(
            graph_1=aligned_layers[la_name], graph_2=aligned_layers[lb_name], alpha=None
        )
        degree_stats.append({(la_name, lb_name): degree_stat})

        partition_stat = correlations.partitions_correlation(
            graph_1=aligned_layers[la_name], graph_2=aligned_layers[lb_name]
        )
        partition_stats.append({(la_name, lb_name): partition_stat})

        edges_stat = correlations.edges_r(aligned_layers[la_name], aligned_layers[lb_name])
        edges_stats.append({(la_name, lb_name): edges_stat})

    return {
        "degree": degree_stats,
        "partition": partition_stats,
        "edges": edges_stats,
    }


def convert_to_correlation_matrix(statistics: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    return {
        "degree": helpers.create_correlation_matrix(statistics["degree"]),
        "partition": helpers.create_correlation_matrix(statistics["partition"]),
        "edges": helpers.create_correlation_matrix(statistics["edges"]),
    }


def save_statistics(statistics: dict[str, pd.DataFrame], net_name: str, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)
    for stat_name, stat_df in statistics.items():
        stat_df.to_csv(out_dir / f"{net_name}_{stat_name}.csv")


def plot_statistics(statistics: dict[str, pd.DataFrame], net_name: str) -> Figure:

    fig, ax = plt.subplots(
        nrows=1, ncols=4, figsize=(18, 6), gridspec_kw={"width_ratios": [33, 33, 33, 1]}
    )
    fig.tight_layout(pad=2.5, rect=(0.05, 0.05, 0.95, 0.95))

    helpers.plot_heatmap(statistics["degree"], ax[0], ax[-1], "degrees")
    helpers.plot_heatmap(statistics["partition"], ax[1], ax[-1], "partitions")
    helpers.plot_heatmap(statistics["edges"], ax[2], ax[-1], "edges R")

    fig.suptitle(net_name)

    return fig


def main(workdir: Path) -> None:

    mode = "destructive"  # "additive" not uset in the paper
    networks = list(NETS_MAPPING.keys())

    workdir.mkdir(exist_ok=True, parents=True)

    pdf = PdfPages(workdir.joinpath(f"correlations.pdf"))
    for net_name in sorted(networks):

        print(net_name)
        net = load_network(net_name, as_tensor=False)
        if net.is_directed(): raise ValueError(
            "Only undirected networks can be processed right now!"
        )

        statistics_raw = compute_statistics(net, mode)
        statistics_df = convert_to_correlation_matrix(statistics_raw)
        n = configuration_model.get_nb_actors(net)
        l = configuration_model.get_nb_layers(net)

        save_statistics(statistics_df, net_name, workdir)
        figure = plot_statistics(statistics_df, f"network: {net_name}, n={n}, l={l}")
        figure.savefig(pdf, format="pdf")
        plt.close(figure)

    pdf.close()


def revive_statistics(csv_path: Path) -> dict[str, str | pd.DataFrame]:
    name_split = csv_path.stem.split("_")
    net_name = "_".join(name_split[:-1])
    corr_type = name_split[-1]
    return {
        "net_name": net_name,
        "corr_type": corr_type,
        "statistics": pd.read_csv(csv_path, index_col=0),
    }


def plot_pretty_statistics(stats_df: dict[str, pd.DataFrame], stats_name: str | None = None) -> None:
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(stats_df) + 1,
        figsize=(4 * len(stats_df), 4.2),
        gridspec_kw={"width_ratios": [*([98 / len(stats_df)] * len(stats_df)), 2]}
    )
    fig.tight_layout(pad=.0, rect=(.0, .0, .97, .95))
    for net_idx, (net_name, net_stat) in enumerate(stats_df.items()):
        helpers.plot_heatmap(
            vis_df=net_stat,
            heatmap_ax=ax[net_idx],
            bar_ax=ax[-1],
            title=NETS_MAPPING[net_name],
            vrange=(-1., 1) if stats_name in {"degree"} else (0., 1.),
        )
    return fig


def pretty_plots(workdir: Path) -> None:
    forbidden_nets = ["toy_network"]
    raw_visualisations = []
    for csv_name in workdir.glob("*.csv"):
        statistics = revive_statistics(csv_name)
        if statistics["net_name"] in forbidden_nets:
            continue
        raw_visualisations.append(statistics)
    
    corr_types = sorted(list({ct["corr_type"] for ct in raw_visualisations}))
    nets = sorted(list({ct["net_name"] for ct in raw_visualisations}))

    pivoted_visualisations = {ct: {n: None for n in nets} for ct in corr_types}
    for rv in raw_visualisations:
        if rv["net_name"] in forbidden_nets:
            continue
        pivoted_visualisations[rv["corr_type"]][rv["net_name"]] = rv["statistics"]
    
    for pv_k, pv_v in pivoted_visualisations.items():
        pdf = PdfPages(workdir.joinpath(f"correlations_{pv_k}.pdf"))
        figure = plot_pretty_statistics(pv_v, pv_k)
        figure.savefig(pdf, format="pdf")
        plt.close(figure)
        pdf.close()


if __name__ == "__main__":
    workdir = Path(__file__).parent.parent.parent / "data/multi_abcd/correlations"
    # main(workdir)
    pretty_plots(workdir)
