import datetime
import warnings
from math import log10

import git
from network_diffusion.utils import fix_random_seed


warnings.filterwarnings(action="ignore", category=FutureWarning)


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2):
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return t_2 - t_1


def get_case_name_base(protocol: str, probab: float, budget: float, ss_name: str, net_name: str) -> str:
    return f"proto-{protocol}--p-{round(probab, 3)}--budget-{budget}--ss-{ss_name}--net-{net_name}"


def get_case_name_rich(
    case_idx: int,
    cases_nb: int,
    rep_idx: int,
    reps_nb: int,
    protocol: str,
    probab: float,
    budget: float, 
    net_name: str,
    ss_name: str,
) -> str:
    return (
        f"repet-{str(rep_idx).zfill(int(log10(reps_nb)+1))}/{reps_nb}--" +
        f"case-{str(case_idx).zfill(int(log10(cases_nb)+1))}/{cases_nb}--" +
        get_case_name_base(protocol, probab, budget, ss_name, net_name)
    )


def get_recent_git_sha() -> str:
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    return git_sha


def set_rng_seed(seed: int) -> None:
    fix_random_seed(seed=seed) # TODO: use it directly from nd once new version is released
