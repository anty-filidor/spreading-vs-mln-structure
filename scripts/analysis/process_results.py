"""Statistical analysis of the results."""

import argparse
import glob
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.notebook import tqdm

from src.aux.results_plotter import ResultsPlotter
from src.aux.results_slicer import ResultsSlicer
from src.loaders.net_loader import load_network

root_path = Path(__file__).resolve().parent.parent.parent


def parse_args(*args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "series",
        help="IDs of series to process results from",
        nargs="?",
        type=list[int],
        default=[0, 1, 2, 3, 4],
    )
    return parser.parse_args(*args)


import sys


def main(series_list: list[int]) -> None:
    print("loading data")
    csv_files = []
    for series in series_list:
        series_csvs = (root_path / f"data/results_raw/series_{series}").glob("**/*.csv")
        csv_files.extend(list(series_csvs))
    results = ResultsSlicer(csv_files)

    workdir = root_path / "data/results_processed"
    workdir.mkdir(exist_ok=True, parents=True)


# # ## Demo

# # In[4]:


# network_name = "l2_course_net_1"
# budget = 15
# protocol = "AND"
# mi_value = 0.10
# ss_method = "random"
# network_graph = load_network(network_name, as_tensor=False)


# # In[ ]:


# r_slice_nml = results.get_slice(
#     protocol=protocol,
#     mi_value=mi_value,
#     seed_budget=budget,
#     network=network_name,
#     ss_method=ss_method,
# )
# r_slice_nml


# # In[ ]:


# r_slice_mds = results.get_slice(
#     protocol=protocol,
#     mi_value=mi_value,
#     seed_budget=budget,
#     network=network_name,
#     ss_method=f"D^{ss_method}",
# )
# r_slice_mds


# # In[ ]:


# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 1.5))  # budget x mi
# slicer_plotter.ResultsPlotter().plot_single_comparison_dynamics(
#     record_mds=results.mean_expositions_rec(r_slice_mds),
#     record_nml=results.mean_expositions_rec(r_slice_nml),
#     actors_nb=results.get_actors_nb(r_slice_mds),
#     mi_value=mi_value,
#     seed_budget=budget,
#     ax=ax
# )


# # In[ ]:


# all_centralities, histogram = results.prepare_centrality(network_graph, "degree")
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 1.5))
# slicer_plotter.ResultsPlotter().plot_single_comparison_centralities(
#     record_mds=r_slice_mds,
#     record_nml=r_slice_nml,
#     all_centralities=all_centralities,
#     hist_centralities=histogram,
#     mi_value=mi_value,
#     seed_budget=budget,
#     ax=ax
# )


# # ## Plot visualisaitons of spreading dynamics to PDF

# # In[ ]:


# print("plotting visualisations of spreading dynamics")
# plotter = slicer_plotter.ResultsPlotter()
# pdf = PdfPages(workdir.joinpath(f"expositions.pdf"))


# # In[ ]:


# for page_idx, page_case in enumerate(plotter.yield_page()):
#     print(page_case)

#     fig, axs = plt.subplots(
#         nrows=len(plotter._seed_budgets_and if page_case[1] == "AND" else plotter._seed_budgets_or),
#         ncols=len(plotter._mi_values),
#         figsize=(15, 20),
#     )

#     for fig_idx, fig_case in tqdm(enumerate(plotter.yield_figure(protocol=page_case[1]))):
#         row_idx = fig_idx // len(axs[0])
#         col_idx = fig_idx % len(axs[1])
#         # print(page_case, fig_case, page_idx, row_idx, col_idx)

#         nml_slice = results.get_slice(
#             protocol=page_case[1],
#             mi_value=fig_case[1],
#             seed_budget=fig_case[0],
#             network=page_case[0],
#             ss_method=page_case[2],
#         )
#         mds_slice = results.get_slice(
#             protocol=page_case[1],
#             mi_value=fig_case[1],
#             seed_budget=fig_case[0],
#             network=page_case[0],
#             ss_method=f"D^{page_case[2]}",
#         )
#         if len(nml_slice) == 0 or len(mds_slice) == 0:
#             plotter.plot_dummy_fig(
#                 mi_value=fig_case[1],
#                 seed_budget=fig_case[0],
#                 ax=axs[row_idx][col_idx],
#             )
#         else:
#             plotter.plot_single_comparison_dynamics(
#                 record_mds=results.mean_expositions_rec(mds_slice),
#                 record_nml=results.mean_expositions_rec(nml_slice),
#                 actors_nb=results.get_actors_nb(nml_slice),
#                 mi_value=fig_case[1],
#                 seed_budget=fig_case[0],
#                 ax=axs[row_idx][col_idx],
#             )
    
#     fig.tight_layout(pad=.5, rect=(0.05, 0.05, 0.95, 0.95))
#     fig.suptitle(f"Network: {page_case[0]}, Protocol: {page_case[1]}, SSM: {page_case[2]}")
#     fig.savefig(pdf, format="pdf")
#     plt.close(fig)

# pdf.close()


# # ## Plot visualisaitons of seed distributions to PDF

# # In[ ]:


# print("plotting visualisations of seed distributions")
# newtorks_centralities = {}
# for network_name in results.raw_df["network"].unique():
#     graph = load_network(network_name, as_tensor=False)
#     degrees = results.prepare_centrality(graph, "degree")
#     neighbourhood_sizes = results.prepare_centrality(graph, "neighbourhood_size")
#     newtorks_centralities[network_name] = {
#         "graph": graph,
#         "degree": {"centr": degrees[0], "hist": degrees[1]},
#         "neighbourhood_size": {"centr": neighbourhood_sizes[0], "hist": neighbourhood_sizes[1]},
#     }


# # In[12]:


# plotter = slicer_plotter.ResultsPlotter()
# pdf = PdfPages(workdir.joinpath(f"distributions.pdf"))


# # In[ ]:


# for page_idx, page_case in enumerate(plotter.yield_page()):
#     print(page_case)

#     centr_name = plotter._centralities[page_case[2]]
#     fig, axs = plt.subplots(
#         nrows=len(plotter._seed_budgets_and if page_case[1] == "AND" else plotter._seed_budgets_or),
#         ncols=len(plotter._mi_values),
#         figsize=(15, 20),
#     )

#     for fig_idx, fig_case in tqdm(enumerate(plotter.yield_figure(protocol=page_case[1]))):
#         row_idx = fig_idx // len(axs[0])
#         col_idx = fig_idx % len(axs[1])
#         # print(page_case, fig_case, page_idx, row_idx, col_idx)

#         nml_slice = results.get_slice(
#             protocol=page_case[1],
#             mi_value=fig_case[1],
#             seed_budget=fig_case[0],
#             network=page_case[0],
#             ss_method=page_case[2],
#         )
#         mds_slice = results.get_slice(
#             protocol=page_case[1],
#             mi_value=fig_case[1],
#             seed_budget=fig_case[0],
#             network=page_case[0],
#             ss_method=f"D^{page_case[2]}",
#         )
#         if len(nml_slice) == 0 or len(mds_slice) == 0:
#             plotter.plot_dummy_fig(
#                 mi_value=fig_case[1],
#                 seed_budget=fig_case[0],
#                 ax=axs[row_idx][col_idx],
#             )
#         else:
#             plotter.plot_single_comparison_centralities(
#                 record_mds=mds_slice,
#                 record_nml=nml_slice,
#                 all_centralities=newtorks_centralities[page_case[0]][centr_name]["centr"],
#                 hist_centralities=newtorks_centralities[page_case[0]][centr_name]["hist"],
#                 mi_value=fig_case[1],
#                 seed_budget=fig_case[0],
#                 ax=axs[row_idx][col_idx],
#             )
    
#     fig.tight_layout(pad=.5, rect=(0.05, 0.05, 0.95, 0.95))
#     fig.suptitle(f"Network: {page_case[0]}, Protocol: {page_case[1]}, SSM: {page_case[2]}")
#     fig.savefig(pdf, format="pdf")
#     plt.close(fig)

# pdf.close()





if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(series_list=args.series)
