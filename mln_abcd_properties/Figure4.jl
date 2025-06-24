using StatsBase
using Plots
using Random
using MLNABCDGraphGenerator
using ABCDGraphGenerator

include("utilities.jl")

#########################
# Correlation tau-sigma #
#########################
seed = 42
iters = 100
n = 1000
vs = 1:n
rhos = -1:0.5:1
qs = 0.6:0.1:1.0
beta = 2.5
min_degree = 5
max_degree = 50

size_mat = length(rhos)
cor_labels = zeros(size_mat, size_mat)
cor_degrees = zeros(size_mat, size_mat)
taus = []
Random.seed!(seed)
for iter in 1:iters
    println("Iteration $(iter)")
    vs_i = [sample(vs, round(Int, q * n), replace=false, ordered=true) for q in qs]
    degs_ordered = [ABCDGraphGenerator.sample_degrees(beta, min_degree, max_degree, length(vs), 1000) for vs in vs_i]
    rankings_taus = MLNABCDGraphGenerator.find_ranking.(vs_i, Ref(n), rhos)
    push!(taus, getindex.(rankings_taus, 1))
    rankings = getindex.(rankings_taus, 2)
    labels_ranked = []
    for layer in zip.(vs_i, rankings)
        tmp = zeros(Int, n)
        for (pos, rank) in layer
            tmp[pos] = rank
        end
        push!(labels_ranked, copy(tmp))
    end
    degs_ranked = []
    for (i, degs) in enumerate(degs_ordered)
        tmp = zeros(Int, n)
        for (j, d) in enumerate(degs)
            tmp[vs_i[i][rankings[i][j]]] = d
        end
        push!(degs_ranked, copy(tmp))
    end
    iter_cor_labels = zeros(size_mat, size_mat)
    iter_cor_degrees = zeros(size_mat, size_mat)
    for i in 1:size_mat
        for j in 1:size_mat
            common_idx = findall(x -> x != 0, degs_ranked[i] .* degs_ranked[j])
            iter_cor_labels[i, j] = corkendall(labels_ranked[i][common_idx], labels_ranked[j][common_idx])
            iter_cor_degrees[i, j] = corkendall(degs_ranked[i][common_idx], degs_ranked[j][common_idx])
        end
    end
    global cor_labels += iter_cor_labels
    global cor_degrees += iter_cor_degrees
end
cor_labels = cor_labels ./ iters
cor_degrees = cor_degrees ./ iters
for i in 1:size_mat
    taus_layer = getindex.(taus, i)
    mean_tau = round(mean(taus_layer), digits=3)
    std_tau = round(std(taus_layer), digits=3)
    println("Layer $(i) | Mean τ: $(mean_tau) | Std τ: $(std_tau)")
end

plt = heatmap(rhos, rhos, cor_labels, c=:Blues, legend=:none)
xticks!(rhos, string.(zip(rhos, qs)))
yticks!(rhos, string.(zip(rhos, qs)))
annotate!(repeat(rhos, inner=5), repeat(rhos, 5), vec(string.(round.(cor_labels, digits=3))))
title!("Mean Kendall τ for Xₐ")
xlabel!("(τ,q)")
ylabel!("(τ,q)")
figsave(plt, "img/rhos_qs_heatmap_labels_kendall.pdf")

plt = heatmap(rhos, rhos, cor_degrees, c=:Blues, legend=:none)
annotate!(repeat(rhos, inner=5), repeat(rhos, 5), vec(string.(round.(cor_degrees, digits=3))))
xticks!(rhos, string.(zip(rhos, qs)))
yticks!(rhos, string.(zip(rhos, qs)))
title!("Mean Kendall τ for degrees")
xlabel!("(τ,q)")
ylabel!("(τ,q)")
figsave(plt, "img/rhos_qs_heatmap_degrees_kendall.pdf")
