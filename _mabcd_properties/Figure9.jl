using StatsBase
using Plots
using Random
using MLNABCDGraphGenerator
using ABCDGraphGenerator
using PyCall
AMI = pyimport("sklearn.metrics").adjusted_mutual_info_score

include("utilities.jl")

############
# r vs AMI #
############
######
# 2D #
######
iter = 10
rs = 0:0.05:1
seed = 42

comms_spec = [(n=1000, min_comm=16, max_comm=32, beta=1.5),
    (n=1000, min_comm=20, max_comm=40, beta=1.5),
    (n=1000, min_comm=20, max_comm=100, beta=1.5),]

amis_all = []
labels = []
Random.seed!(seed)
for spec in comms_spec
    c1 = ABCDGraphGenerator.sample_communities(spec.beta, spec.min_comm, spec.max_comm, spec.n, 1000)
    c2 = ABCDGraphGenerator.sample_communities(spec.beta, spec.min_comm, spec.max_comm, spec.n, 1000)
    x = MLNABCDGraphGenerator.sample_points(spec.n, 2)
    a1 = MLNABCDGraphGenerator.assign_points(x, c1)
    a2 = MLNABCDGraphGenerator.assign_points(x, c2)
    amis = []
    for r in rs
        println("Processing r: $(r) for spec $(spec)")
        iter_amis = []
        for i in 1:iter
            shuffled_flat_a1 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a1))
            shuffled_flat_a2 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a2))
            push!(iter_amis, AMI(shuffled_flat_a1, shuffled_flat_a2))
        end
        push!(amis, (mean(iter_amis), std(iter_amis)))
    end
    push!(amis_all, amis)
    push!(labels, "s=$(spec.min_comm) S=$(spec.max_comm)")
end

plt = plot_cluster_stats(rs,
    [getindex.(g, 1) for g in amis_all],
    [getindex.(g, 2) for g in amis_all],
    labels,
    2)
figsave(plt, "img/ami_r_lineplot_2D_compared_to_r.pdf")

######
# 1D #
######

amis_all = []
labels = []
Random.seed!(seed)
for spec in comms_spec
    c1 = ABCDGraphGenerator.sample_communities(spec.beta, spec.min_comm, spec.max_comm, spec.n, 1000)
    c2 = ABCDGraphGenerator.sample_communities(spec.beta, spec.min_comm, spec.max_comm, spec.n, 1000)
    x = MLNABCDGraphGenerator.sample_points(spec.n, 1)
    a1 = assign_points_1d(vec(x), c1)
    a2 = assign_points_1d(vec(x), c2)
    amis = []
    for r in rs
        println("Processing r: $(r) for spec $(spec)")
        iter_amis = []
        iter_aris = []
        for i in 1:iter
            shuffled_flat_a1 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a1))
            shuffled_flat_a2 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a2))
            push!(iter_amis, AMI(shuffled_flat_a1, shuffled_flat_a2))
        end
        push!(amis, (mean(iter_amis), std(iter_amis)))
    end
    push!(amis_all, amis)
    push!(labels, "s=$(spec.min_comm) S=$(spec.max_comm)")
end

plt = plot_cluster_stats(rs,
    [getindex.(g, 1) for g in amis_all],
    [getindex.(g, 2) for g in amis_all],
    labels,
    1)
figsave(plt, "img/ami_r_lineplot_1D_compared_to_r.pdf")
