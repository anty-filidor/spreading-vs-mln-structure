"""Statistical analysis of the results."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.aux.results_plotter import ResultsPlotter
from src.aux.results_slicer import ResultsSlicer


root_path = Path(__file__).resolve().parent.parent.parent


def parse_args(*args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "series",
        help="IDs of series to process results from",
        nargs="?",
        type=list[str],
        # default=["1", "2", "3", "4", "5"],
        # default=["1", "6", "7", "8", "9"],
        default=["1", "10", "11", "12", "13"],
    )
    parser.add_argument(
        "baseline_type",
        help="Type of the network to be useda as a baseline run",
        nargs="?",
        type=str,
        default="series_1",

    )
    return parser.parse_args(*args)


def main(series_list: list[str], baseline_type: str) -> None:
    print("loading data")
    csv_files = []
    for series in series_list:
        series_csvs = (root_path / f"data/results_raw/series_{series}").glob("**/*.csv")
        csv_files.extend(list(series_csvs))
    results = ResultsSlicer(results_paths=csv_files, baseline_type=baseline_type)

    # create out dir
    workdir = root_path / f"data/results_processed/{'_'.join([s for s in series_list])}"
    workdir.mkdir(exist_ok=True, parents=True)

    # analyse the results
    out_pdf = PdfPages(workdir.joinpath(f"expositions.pdf"))
    out_csv = []
    for (protocol, probab, seed_budget, ss_method) in results.get_combinations():
        case_name = f"δ={protocol}, π={probab}, s={seed_budget}, φ={ss_method}"
        print(case_name)

        # for each case obtain partial raw results
        records_experiments = {}
        for net_type in results.get_net_types():
            results_slice = results.get_slice(
                protocol=protocol,
                probab=probab,
                seed_budget=seed_budget,
                ss_method=ss_method,
                net_type=net_type,
            )
            if len(results_slice) == 0:
                print(f"\tno results found for {net_type}")
                continue

            # compute mean expositions for the visualisation
            records_experiments[net_type] = results.mean_expositions_rec(results_slice)
        
            # update table with average metrics
            out_csv.append(
                {
                    "protocol": protocol,
                    "probab": probab,
                    "seed_budget": seed_budget,
                    "ss_method": ss_method,
                    "net_type": net_type,
                    "gain_avg": results_slice["gain"].mean(),
                    "gain_std": results_slice["gain"].std(),
                    "area_avg": results_slice["area"].mean(),
                    "area_std": results_slice["area"].std(),
                }
            )

        # plot spreading dynamics        
        if len(records_experiments) == 0:
            continue
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5))
        ResultsPlotter().plot_single_comparison_dynamics(
            records_experiments=records_experiments,
            baseline_key=results.baseline_type,
            title=case_name,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)

    # save results
    out_pdf.close()
    pd.DataFrame(out_csv).to_csv(workdir / "metrics.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(series_list=args.series, baseline_type=args.baseline_type)
