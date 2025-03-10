"""A Python wrapper to the MLNABCDGraphGenerator Julia package."""

from dataclasses import dataclass

from juliacall import JuliaError
from juliacall import Main as jl


@dataclass
class MLNConfig:
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

    def __post_init__(self) -> None:
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

        # Load config
        config = jl.MLNABCDGraphGenerator.MLNConfig(
            config.seed,
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

    mln_config = MLNConfig(
        seed=42,
        n=1000,
        edges_cor="/Users/michal/Development/MLNABCDGraphGenerator.jl/utils/edges_cor_matrix.csv",
        layer_params= "/Users/michal/Development/MLNABCDGraphGenerator.jl/utils/layer_params.csv",
        d_max_iter=1000,
        c_max_iter=1000,
        t=100,
        eps=0.01,
        d=2,
        edges_filename="./edgelist.dat",
        communities_filename="./communities.dat",
    )

    MLNABCDGraphGenerator()(config=mln_config)
