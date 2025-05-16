"""Main runner of the simulator."""

import yaml
from typing import Any

from tqdm import tqdm

from src import params_handler, result_handler, utils
from src.simulator import ranking_runner


def run_experiments(config: dict[str, Any]) -> None:

    # load networks, initialise ssms and evaluated parameter space
    nets = params_handler.load_networks(
        networks=config["parameter_space"]["networks"],
        device=config["run"]["device"]
    )
    ssms = params_handler.load_seed_selectors(config["parameter_space"]["ss_methods"])
    p_space = params_handler.get_parameter_space(
        protocols=config["parameter_space"]["protocols"],
        probabs=config["parameter_space"]["probabs"],
        seed_budgets=config["parameter_space"]["seed_budgets"],
        ss_methods=config["parameter_space"]["ss_methods"],
        networks=[(net.n_type, net.n_name) for net in nets],
    )

    # get parameters of the simulator
    ranking_path = config["io"].get("ranking_path")
    repetitions = config["simulator"]["repetitions"]
    rng_seed = "_"if config["run"].get("rng_seed") is None else config["run"]["rng_seed"]

    # prepare output directories and determine how to store results
    out_dir = params_handler.create_out_dir(config["io"]["out_dir"])
    rnk_dir = out_dir / result_handler.RANKINGS_DIR
    rnk_dir.mkdir(exist_ok=True, parents=True)
    compress_to_zip = config["io"]["compress_to_zip"]

    # save the config
    config["git_sha"] = utils.get_recent_git_sha()
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # get a start time
    start_time = utils.get_current_time()
    print(f"\nExperiments started at {start_time}")

    # repeat main loop for given number of times
    for rep in range(1, repetitions + 1):
        print(f"\nRepetition {rep}/{repetitions}\n")
        rep_results = []
        ver = f"{rng_seed}_{rep}"

        # for each network ans ss method compute a ranking and save it
        rankings = params_handler.compute_rankings(
            seed_selectors=ssms,
            networks=nets,
            out_dir=rnk_dir,
            version=ver,
            ranking_path=ranking_path,
        )

        # start simulations
        p_bar = tqdm(p_space, desc="", leave=False, colour="green")
        for idx, investigated_case in enumerate(p_bar):
            proto, budget, p, net_type_name, ss_method = investigated_case
            try:
                net = [
                    net for net in nets if 
                    net.n_type == net_type_name[0] and net.n_name == net_type_name[1]
                ][0]
                p_bar.set_description_str(
                    utils.get_case_name_rich(
                        rep_idx=rep,
                        reps_nb=repetitions,
                        case_idx=idx,
                        cases_nb=len(p_bar),
                        protocol=proto,
                        probab=p,
                        budget=budget[1],
                        net_name=net.rich_name,
                        ss_name=ss_method,
                    )
                )
                investigated_case_results = ranking_runner.handle_step(
                    proto=proto, 
                    p=p,
                    budget=budget,
                    ss_method=ss_method,
                    net=net,
                    ranking=rankings[(net.rich_name, ss_method)],
                    max_epochs_num=config["simulator"]["max_epochs_num"],
                )
                rep_results.extend(investigated_case_results)
            except BaseException as e:
                base_name = utils.get_case_name_base(proto, p, budget[1], ss_method, net.rich_name)
                print(f"\nExperiment failed for case: {base_name}--ver-{ver}")
                raise e
        
        # aggregate results for given repetition number and save them to a csv file
        result_handler.save_results(rep_results, out_dir / f"results--ver-{ver}.csv")

    # compress global logs and config
    if compress_to_zip:
        result_handler.zip_detailed_logs([rnk_dir], rm_logged_dirs=True)

    finish_time = utils.get_current_time()
    print(f"\nExperiments finished at {finish_time}")
    print(f"Experiments lasted {utils.get_diff_of_times(start_time, finish_time)} minutes")
