using Plots
using Random
using StatsBase
using ABCDGraphGenerator
using MLNABCDGraphGenerator

include("utilities.jl")

######################
# Reference layer 2D #
######################
n = 1000
β = 1.5
max_c = round(Int, sqrt(n))
min_c = round(Int, 0.5 * max_c)
max_iter = 1000
d = 2
seed = 42

# All agents active
Random.seed!(seed)
c = ABCDGraphGenerator.sample_communities(β, min_c, max_c, n, max_iter)
x = MLNABCDGraphGenerator.sample_points(n, d)
a = MLNABCDGraphGenerator.assign_points(x, c)
plt = plot_reference_space_2d(x, a, collect(1:n))
figsave(plt, "img/reference_layer.pdf")

# Half agents inactive
q = 0.5
min_c = 25
max_c = 50

n_q = Int(q * n)
Random.seed!(seed)
active = sample(1:n, n_q, replace=false, ordered=true)
c = ABCDGraphGenerator.sample_communities(β, min_c, max_c, n_q, max_iter)
x = MLNABCDGraphGenerator.sample_points(n, d)
a = MLNABCDGraphGenerator.assign_points(x[active, :], c)
plt = plot_reference_space_2d(x, a, active)
figsave(plt, "img/reference_layer_50perc_active.pdf")