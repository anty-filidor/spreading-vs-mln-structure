"""A script with functions facilitating saving the results."""

import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


DET_LOGS_DIR = "detailed_logs"
RANKINGS_DIR = "rankings"


@dataclass(frozen=True)
class SimulationPartialResult:
    seed_ids: str  # IDs of actors that were seeds aggr. into string (sep. by ;)
    gain: float  # gain obtained using this seed set
    area: float | None  # area under normalised expositions curve reflecting diffusion dynamics
    simulation_length: int  # nb. of simulation steps
    seed_nb: int  # nb. of actors that were seeds
    exposed_nb: int  # nb. of active actors at the end of the simulation
    unexposed_nb: int  # nb. of actors that remained inactive
    expositions_rec: str  # record of new activations aggr. into string (sep. by ;)


@dataclass(frozen=True)
class SimulationFullResult(SimulationPartialResult):
    network_type: str  # network's type
    network_name: str  # network's name
    ss_method: str  # seed selection method's name
    seed_budget: float  # a value of the maximal seed budget
    protocol: str  # protocols's (aggragation function) name
    probab: float  # a value of the activation probability

    @classmethod
    def enhance_SPR(
        cls,
        SPR: SimulationPartialResult,
        network_type: str,
        network_name: str,
        ss_method: str,
        seed_budget: float,
        protocol: str,
        probab: float,
    ) -> "SimulationFullResult":
        return cls(
            seed_ids=SPR.seed_ids,
            gain=SPR.gain,
            area=SPR.area,
            simulation_length=SPR.simulation_length,
            seed_nb=SPR.seed_nb,
            exposed_nb=SPR.exposed_nb,
            unexposed_nb=SPR.unexposed_nb,
            expositions_rec=SPR.expositions_rec,
            network_type=network_type,
            network_name=network_name,
            ss_method=ss_method,
            seed_budget=seed_budget,
            protocol=protocol,
            probab=probab,
        )


def save_results(result_list: list[SimulationFullResult], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in result_list]
    pd.DataFrame(me_dict_all).to_csv(out_path, index=False)


def zip_detailed_logs(logged_dirs: list[Path], rm_logged_dirs: bool = True) -> None:
    if len(logged_dirs) == 0:
        print("No directories provided to create archive from.")
        return
    for dir_path in logged_dirs:
        shutil.make_archive(logged_dirs[0].parent / dir_path.name, "zip", root_dir=str(dir_path))
    if rm_logged_dirs:
        for dir_path in logged_dirs:
            shutil.rmtree(dir_path)
    print(f"Compressed detailed logs")
