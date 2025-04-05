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
        type=list[int],
        default=[0, 1, 2, 3, 4],
    )
    return parser.parse_args(*args)


def main(series_list: list[int]) -> None:
    print("loading data")
    csv_files = []
    for series in series_list:
        series_csvs = (root_path / f"data/results_raw/series_{series}").glob("**/*.csv")
        csv_files.extend(list(series_csvs))
    results = ResultsSlicer(csv_files)

    # create out dir
    workdir = root_path /f"data/results_processed/{'_'.join([str(s) for s in series_list])}"
    workdir.mkdir(exist_ok=True, parents=True)

    # analyse results
    out_pdf = PdfPages(workdir.joinpath(f"expositions.pdf"))
    out_csv = []
    for (protocol, mi_value, seed_budget, ss_method) in results.get_combinations():
        case_name = f"δ={protocol}, μ={mi_value}, s={seed_budget}, φ={ss_method}"
        print(case_name)

        # for each case obtain partial raw results
        record_baseline = None
        records_experiments = {}
        for net_type in results.get_net_types():
            results_slice = results.get_slice(
                protocol=protocol,
                mi_value=mi_value,
                seed_budget=seed_budget,
                ss_method=ss_method,
                net_type=net_type,
            )
            if len(results_slice) == 0:
                print(f"no results found for {net_type}")
                continue

            # compute mean expositions for the visualisation
            if "series_0" in net_type:
                record_baseline = results.mean_expositions_rec(results_slice)
            else:
                records_experiments[net_type] = results.mean_expositions_rec(results_slice)
        
            # update table with average metrics
            out_csv.append(
                {
                    "protocol": protocol,
                    "mi_value": mi_value,
                    "seed_budget": seed_budget,
                    "ss_method": ss_method,
                    "net_type": net_type,
                    "gain_avg": results_slice["gain"].mean(),
                    "gain_std": results_slice["gain"].std(),
                    "auc_avg": results_slice["auc"].mean(),
                    "auc_std": results_slice["auc"].std(),
                }
            )

        # plot spreading dynamics        
        if record_baseline is None:
            continue
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5))
        ResultsPlotter().plot_single_comparison_dynamics(
            record_baseline=record_baseline,
            records_experiments=records_experiments,
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
    main(series_list=args.series)
