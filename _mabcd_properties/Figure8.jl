using StatsBase
using Plots
using Random
using MLNABCDGraphGenerator
using ABCDGraphGenerator
using PyCall
AMI = pyimport("sklearn.metrics").adjusted_mutual_info_score

include("utilities.jl")

###########################
# q₁ vs q₂ AMI only active#
###########################
n = 1000
iter = 100
qs = 0.5:0.025:1
r = 1.0
seed = 42
d = 2

######
# 2D #
######
Random.seed!(seed)
x = MLNABCDGraphGenerator.sample_points(n, 2)
amis_q = []
for q1 in qs
    for q2 in qs
        println("Processing q1: $(q1) and q2: $(q2)")
        iter_amis = []
        for i in 1:iter
            c1_size = round(Int, q1 * n)
            c2_size = round(Int, q2 * n)
            active_nodes1 = sample(1:n, c1_size, replace=false, ordered=true)
            active_nodes2 = sample(1:n, c2_size, replace=false, ordered=true)
            c1 = ABCDGraphGenerator.sample_communities(1.5, 8, 32, c1_size, 1000)
            c2 = ABCDGraphGenerator.sample_communities(1.5, 8, 32, c2_size, 1000)
            a1 = MLNABCDGraphGenerator.assign_points(x[active_nodes1, :], c1)
            a2 = MLNABCDGraphGenerator.assign_points(x[active_nodes2, :], c2)
            shuffled_flat_a1 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a1))
            shuffled_flat_a2 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a2))
            full_a1 = zeros(Int, n)
            full_a2 = zeros(Int, n)
            for (i, c) in enumerate(shuffled_flat_a1)
                full_a1[active_nodes1[i]] = c
            end
            for (i, c) in enumerate(shuffled_flat_a2)
                full_a2[active_nodes2[i]] = c
            end
            common_idx = findall(x -> x != 0, full_a1 .* full_a2)
            push!(iter_amis, AMI(full_a1[common_idx], full_a2[common_idx]))
        end
        push!(amis_q, (mean(iter_amis), std(iter_amis), q1, q2))
    end
end

reshaped_amis_q = reshape(getindex.(amis_q, 1), 21, 21)
plt = heatmap(qs, qs, reshaped_amis_q, ticks=qs[1:2:end])
title!("Mean AMI 2D")
xlabel!("q₁")
ylabel!("q₂")
figsave(plt, "img/ami_q_heatmap_2D_only_active.pdf")

######
# 1D #
######
Random.seed!(seed)
x1 = MLNABCDGraphGenerator.sample_points(n, 1)
amis_1d_q = []
for q1 in qs
    for q2 in qs
        println("Processing q1: $(q1) and q2: $(q2)")
        iter_amis = []
        for i in 1:iter
            c1_size = round(Int, q1 * n)
            c2_size = round(Int, q2 * n)
            active_nodes1 = sample(1:n, c1_size, replace=false, ordered=true)
            active_nodes2 = sample(1:n, c2_size, replace=false, ordered=true)
            c1 = ABCDGraphGenerator.sample_communities(1.5, 8, 32, c1_size, 1000)
            c2 = ABCDGraphGenerator.sample_communities(1.5, 8, 32, c2_size, 1000)
            a1 = assign_points_1d(x1[active_nodes1], c1)
            a2 = assign_points_1d(x1[active_nodes2], c2)
            shuffled_flat_a1 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a1))
            shuffled_flat_a2 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a2))
            full_a1 = zeros(Int, n)
            full_a2 = zeros(Int, n)
            for (i, c) in enumerate(shuffled_flat_a1)
                full_a1[active_nodes1[i]] = c
            end
            for (i, c) in enumerate(shuffled_flat_a2)
                full_a2[active_nodes2[i]] = c
            end
            common_idx = findall(x -> x != 0, full_a1 .* full_a2)
            push!(iter_amis, AMI(full_a1[common_idx], full_a2[common_idx]))
        end
        push!(amis_1d_q, (mean(iter_amis), std(iter_amis), q1, q2))
    end
end

reshaped_amis_1d_q = reshape(getindex.(amis_1d_q, 1), 21, 21)
plt = heatmap(qs, qs, reshaped_amis_1d_q, ticks=qs[1:2:end])
title!("Mean AMI 1D")
xlabel!("q₁")
ylabel!("q₂")
figsave(plt, "img/ami_q_heatmap_1D_only_active.pdf")
