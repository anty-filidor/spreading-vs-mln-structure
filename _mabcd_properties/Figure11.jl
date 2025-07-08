using StatsBase
using Plots
using Random
using MLNABCDGraphGenerator
using ABCDGraphGenerator
using DelimitedFiles
using PyCall
AMI = pyimport("sklearn.metrics").adjusted_mutual_info_score

include("utilities.jl")

###############
# 5 layers AMI#
###############
######
# 2D #
######

iters = 100
n = 1000
rs = 0:0.25:1
qs = 0.6:0.1:1.0
βs = 1.1:0.2:1.9
ss = 8:16:72
Ss = 32:16:96
Random.seed!(42)
x = MLNABCDGraphGenerator.sample_points(n, 2)
amis_mat = zeros(5, 5)
vs = 1:n
for iter in 1:iters
    println("Iteration $(iter)")
    vs_i = [sample(vs, round(Int, q * n), replace=false, ordered=true) for q in qs]
    coms = [ABCDGraphGenerator.sample_communities(βs[i], ss[i], Ss[i], length(vs_i[i]), 1000) for i in eachindex(vs_i)]
    as = [MLNABCDGraphGenerator.assign_points(x[vs_i[i], :], coms[i]) for i in eachindex(vs_i)]
    shuffled_flat_as = [flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(rs[i], as[i])) for i in eachindex(vs_i)]
    full_as = [zeros(Int, n) for i in eachindex(vs_i)]
    for i in eachindex(vs_i)
        for (j, c) in enumerate(shuffled_flat_as[i])
            full_as[i][vs_i[i][j]] = c
        end
    end
    iter_ami_mat = zeros(5, 5)
    for i in 1:5
        for j in 1:5
            common_idx = findall(x -> x != 0, full_as[i] .* full_as[j])
            iter_ami_mat[i, j] = AMI(full_as[i][common_idx], full_as[j][common_idx])
        end
    end
    global amis_mat += iter_ami_mat
end
amis_mat = abs.(amis_mat ./ iters)

plt = heatmap(rs, rs, amis_mat, c=:Blues, legend=:none)
xticks!(rs, string.(zip(βs, ss, Ss)) .* '\n' .* string.(zip(qs, rs)))
yticks!(rs, string.(zip(βs, ss, Ss)) .* '\n' .* string.(zip(qs, rs)))
annotate!(repeat(rs, inner=5), repeat(rs, 5), vec(string.(round.(amis_mat, digits=3))))
title!("Mean AMI 2D")
xlabel!("(β,s,S)(q,r)")
ylabel!("(β,s,S)(q,r)")
figsave(plt, "img/beta_s_S_q_r_heatmap_2D.pdf")

######
# 1D #
######

Random.seed!(42)
x1 = MLNABCDGraphGenerator.sample_points(n, 1)
amis_mat_1d = zeros(5, 5)
for iter in 1:iters
    println("Iteration $(iter)")
    vs_i = [sample(vs, round(Int, q * n), replace=false, ordered=true) for q in qs]
    coms = [ABCDGraphGenerator.sample_communities(βs[i], ss[i], Ss[i], length(vs_i[i]), 1000) for i in eachindex(vs_i)]
    as = [assign_points_1d(x1[vs_i[i]], coms[i]) for i in eachindex(vs_i)]
    shuffled_flat_as = [flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(rs[i], as[i])) for i in eachindex(vs_i)]
    full_as = [zeros(Int, n) for i in eachindex(vs_i)]
    for i in eachindex(vs_i)
        for (j, c) in enumerate(shuffled_flat_as[i])
            full_as[i][vs_i[i][j]] = c
        end
    end
    iter_ami_mat = zeros(5, 5)
    for i in 1:5
        for j in 1:5
            common_idx = findall(x -> x != 0, full_as[i] .* full_as[j])
            iter_ami_mat[i, j] = AMI(full_as[i][common_idx], full_as[j][common_idx])
        end
    end
    global amis_mat_1d += iter_ami_mat
end
amis_mat_1d = abs.(amis_mat_1d ./ iters)

plt = heatmap(rs, rs, amis_mat_1d, c=:Blues, legend=:none)
xticks!(rs, string.(zip(βs, ss, Ss)) .* '\n' .* string.(zip(qs, rs)))
yticks!(rs, string.(zip(βs, ss, Ss)) .* '\n' .* string.(zip(qs, rs)))
annotate!(repeat(rs, inner=5), repeat(rs, 5), vec(string.(round.(amis_mat_1d, digits=3))))
title!("Mean AMI 1D")
xlabel!("(β,s,S)(q,r)")
ylabel!("(β,s,S)(q,r)")
figsave(plt, "img/beta_s_S_q_r_heatmap_1D.pdf")
