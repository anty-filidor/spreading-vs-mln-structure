# from julia.api import Julia
# # jl = Julia(compiled_modules=False)

# from julia import Main
# Main.println("I'm printing from a Julia function!")

from juliacall import Main as jl

# jl.Pkg.add(url="https://github.com/bkamins/ABCDGraphGenerator.jl")
# jl.Pkg.add(url="https://github.com/KrainskiL/MLNABCDGraphGenerator.jl")
jl.seval("using MLNABCDGraphGenerator")

edges_cor_path = "/Users/michal/Development/MLNABCDGraphGenerator.jl/utils/edges_cor_matrix.csv"
layer_params_path = "/Users/michal/Development/MLNABCDGraphGenerator.jl/utils/layer_params.csv"
edges_filename = "./edgelist.dat"
communities_filename = "./communities.dat"

config = jl.MLNABCDGraphGenerator.MLNConfig(
    42,  # seed
    1000,  # n
    edges_cor_path,  # edges_cor
    layer_params_path,  # layer_params
    1000,  # d_max_iter
    1000,  # c_max_iter
    100,  # t
    0.01,  # eps
    2,  # d
    edges_filename,  # edges_filename
    communities_filename,  # communities_filename
)
# config = jl.MLNABCDGraphGenerator.parse_config(filename)

#Active nodes
active_nodes = jl.MLNABCDGraphGenerator.generate_active_nodes(config)
#Degree Sequences
degrees = jl.MLNABCDGraphGenerator.generate_degrees(config, active_nodes, False)
#Sizes of communities
com_sizes, coms = jl.MLNABCDGraphGenerator.generate_communities(config, active_nodes)
#Generate ABCD graphs
edges = jl.MLNABCDGraphGenerator.generate_abcd(config, degrees, com_sizes, coms)
#Map nodes and communities to agents
edges = jl.MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
coms = jl.MLNABCDGraphGenerator.map_communities_to_agents(config.n, coms, active_nodes)
#Adjust edges correlation
edges_rewired = jl.MLNABCDGraphGenerator.adjust_edges_correlation(config, edges, coms, active_nodes, False, False)
#Save edges to file
jl.MLNABCDGraphGenerator.write_edges(config, edges_rewired)
#Save communities to file
jl.MLNABCDGraphGenerator.write_communities(config, coms)















jl.println("Hello from Julia!")
# Hello from Julia!
x = jl.rand(range(10), 3, 5)
x._jl_display()
# 3×5 Matrix{Int64}:
#  8  1  7  0  6
#  9  2  1  4  0
#  1  8  5  4  0
import numpy
numpy.sum(x, axis=0)
# array([18, 11, 13,  8,  6], dtype=int64)
