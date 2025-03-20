"""A Python wrapper to the MLNABCDGraphGenerator Julia package."""

from dataclasses import asdict, dataclass

from juliacall import JuliaError
from juliacall import Main as jl
import numpy as np
import pandas as pd
import yaml

@dataclass
class MLNConfig:
    """
    A wrapper for class for class for jl.MLNABCDGraphGenerator.MLNConfig.

    TODO: we can get rid of storing a part of the config in files (see commented out code and:
    https://github.com/KrainskiL/MLNABCDGraphGenerator.jl/blob/main/src/auxiliary.jl#L19)
    """
    seed: int
    n: int
    edges_cor: str
    layer_params: str
    d_max_iter: int
    c_max_iter: int
    t: int
    eps: float
    d: int
    edges_filename: str
    communities_filename: str
    # l: int
    # qs: list[float]
    # ns: list[int]
    # taus: list[float]
    # rs: list[float]
    # gammas: list[float]
    # d_mins: list[int]
    # d_maxs: list[int]
    # betas: list[float]
    # c_mins: list[int]
    # c_maxs: list[int]
    # xis: list[float]
    # skip_edges_correlation: bool
    # edges_cor_matrix: np.ndarray

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(seed=self.seed)
        assert isinstance(self.seed, int)
        assert isinstance(self.n, int)
        assert isinstance(self.edges_cor, str)
        assert isinstance(self.layer_params, str)
        assert isinstance(self.d_max_iter, int)
        assert isinstance(self.c_max_iter, int)
        assert isinstance(self.t, int)
        assert isinstance(self.d, int)
        assert isinstance(self.eps, float)
        assert isinstance(self.d, int)
        assert isinstance(self.edges_filename, str)
        assert isinstance(self.communities_filename, str)
        # assert isinstance(self.l, int)
        # assert isinstance(self.qs, list)
        # assert all([isinstance(q, float) for q in self.qs])
        # assert isinstance(self.ns, list)
        # assert all([isinstance(n, int) for n in self.ns])
        # assert isinstance(self.taus, list)
        # assert all([isinstance(tau, float) for tau in self.taus])
        # assert isinstance(self.rs, list)
        # assert all([isinstance(r, float) for r in self.rs])
        # assert isinstance(self.gammas, list)
        # assert all([isinstance(gamma, float) for gamma in self.gammas])
        # assert isinstance(self.d_mins, list)
        # assert all([isinstance(d_min, int) for d_min in self.d_mins])
        # assert isinstance(self.d_maxs, list)
        # assert all([isinstance(d_max, int) for d_max in self.d_maxs])
        # assert isinstance(self.betas, list)
        # assert all([isinstance(beta, float) for beta in self.betas])
        # assert isinstance(self.c_mins, list)
        # assert all([isinstance(c_min, int) for c_min in self.c_mins])
        # assert isinstance(self.c_maxs, list)
        # assert all([isinstance(c_max, int) for c_max in self.c_maxs])
        # assert isinstance(self.xis, list)
        # assert all([isinstance(xi, float) for xi in self.xis])
        # assert isinstance(self.skip_edges_correlation, bool)
        # assert isinstance(self.edges_cor_matrix, np.ndarray)

    # @classmethod
    # def from_csvs(
    #     cls,
    #     seed: int,
    #     n: int,
    #     edges_cor: str,
    #     layer_params: str,
    #     d_max_iter: int,
    #     c_max_iter: int,
    #     t: int,
    #     eps: float,
    #     d: int,
    #     edges_filename: str,
    #     communities_filename: str,
    # ) -> "MLNConfig":
    #     layer_params_df = pd.read_csv(layer_params)
    #     edges_cor_arr = pd.read_csv(edges_cor, index_col=0).astype(float).to_numpy()
    #     return cls(
    #         seed=seed,
    #         n=n,
    #         edges_cor=edges_cor,
    #         layer_params=layer_params,
    #         d_max_iter=d_max_iter,
    #         c_max_iter=c_max_iter,
    #         t=t,
    #         eps=eps,
    #         d=d,
    #         edges_filename=edges_filename,
    #         communities_filename=communities_filename,
    #         l=len(layer_params_df),
    #         qs=layer_params_df["q"].tolist(),
    #         ns=(n * layer_params_df["q"]).astype(int).tolist(),
    #         taus=layer_params_df["tau"].tolist(),
    #         rs=layer_params_df["r"].tolist(),
    #         gammas=layer_params_df["gamma"].tolist(),
    #         d_mins=layer_params_df["delta"].astype(int).tolist(),
    #         d_maxs=layer_params_df["Delta"].astype(int).tolist(),
    #         betas=layer_params_df["beta"].tolist(),
    #         c_mins=layer_params_df["s"].astype(int).tolist(),
    #         c_maxs=layer_params_df["S"].astype(int).tolist(),
    #         xis=layer_params_df["xi"].tolist(),
    #         skip_edges_correlation=False,
    #         edges_cor_matrix=edges_cor_arr,
    #     )
    
    def to_yaml(self, filename: str):
        with open(filename, "w") as file:
            yaml.dump(asdict(self), file, default_flow_style=False)

    @staticmethod
    def from_yaml(filename: str):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        return MLNConfig(**data)


class MLNABCDGraphGenerator:
    """A wrapper class for jl.MLNABCDGraphGenerator."""

    @staticmethod
    def install_julia_dependencies():
        jl.Pkg.add(url="https://github.com/bkamins/ABCDGraphGenerator.jl")
        jl.Pkg.add(url="https://github.com/KrainskiL/MLNABCDGraphGenerator.jl")

    def __call__(self, config: MLNConfig) -> None:
        try:
            jl.seval("using MLNABCDGraphGenerator")
        except JuliaError:
            self.install_julia_dependencies()

        # Load config. Since julia is called each time as a new process, we use a following
        # workaround to generate random, yet repetitive as a sequence, results
        config = jl.MLNABCDGraphGenerator.MLNConfig(
            int(config._rng.random() * 1000),
            config.n,
            config.edges_cor,
            config.layer_params,
            config.d_max_iter,
            config.c_max_iter,
            config.t,
            config.eps,
            config.d,
            config.edges_filename,
            config.communities_filename,
            # config.l,
            # config.qs,
            # config.ns,
            # config.taus,
            # config.rs,
            # config.gammas,
            # config.d_mins,
            # config.d_maxs,
            # config.betas,
            # config.c_mins,
            # config.c_maxs,
            # config.xis,
            # config.skip_edges_correlation,
            # config.edges_cor_matrix,
        )

        # Active nodes
        active_nodes = jl.MLNABCDGraphGenerator.generate_active_nodes(config)

        # Degree Sequences
        degrees = jl.MLNABCDGraphGenerator.generate_degrees(config, active_nodes, False)

        # Sizes of communities
        com_sizes, coms = jl.MLNABCDGraphGenerator.generate_communities(config, active_nodes)

        # Generate ABCD graphs
        edges = jl.MLNABCDGraphGenerator.generate_abcd(config, degrees, com_sizes, coms)

        # Map nodes and communities into agents
        edges = jl.MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
        coms = jl.MLNABCDGraphGenerator.map_communities_to_agents(config.n, coms, active_nodes)

        # Adjust edges correlation
        edges_rewired = jl.MLNABCDGraphGenerator.adjust_edges_correlation(
            config, edges, coms, active_nodes, False, False
        )

        # Save edges to file
        jl.MLNABCDGraphGenerator.write_edges(config, edges_rewired)

        # Save communities to file
        jl.MLNABCDGraphGenerator.write_communities(config, coms)


if __name__ == "__main__":

    # load from code
    mln_config = MLNConfig(
        seed=42,
        n=1000,
        edges_cor="scripts/configs/example_generate/edges_cor.csv",
        layer_params="scripts/configs/example_generate/layer_params.csv",
        d_max_iter=1000,
        c_max_iter=1000,
        t=100,
        eps=0.01,
        d=2,
        edges_filename="./edges.dat",
        communities_filename="./communities.dat",
    )

    # # or from files
    mln_config = MLNConfig.from_yaml("scripts/configs/example_generate/mln_config.yaml")

    # then, generate a network
    MLNABCDGraphGenerator()(config=mln_config)

