using Plots
using ABCDGraphGenerator
using MLNABCDGraphGenerator
using Random
using StatsBase

function plot_reference_space_2d(x::Matrix{Float64}, a::Vector{Vector{Int}}, active::Vector{Int})
    n = size(x)[1]
    active_mapping = Dict(enumerate(active))
    all_active = n == length(active)
    plt = plot(lims=(-1.1, 1.1), size=(400, 400))
    if !all_active
        inactive = setdiff(1:n, active)
        scatter!(plt, x[inactive, 1], x[inactive, 2], color="gray", markersize=4, markerstrokewidth=0, label=nothing, lims=(-1.1, 1.1), aspect_ratio=:equal)
    end
    for z in a
        rows = getindex.(Ref(active_mapping), z)
        scatter!(plt, x[rows, 1], x[rows, 2], lims=(-1.1, 1.1), aspect_ratio=:equal, label=nothing)
    end
    return plt
end

function powerlaw_cdfs(seqs::Vector{Vector{Int}})
    cdfs = Vector{Float64}[]
    ks = Vector{Int}[]
    for seq in seqs
        dmin, dmax = extrema(seq)
        cdf = Float64[]
        k = Int[]
        for d in dmin:dmax
            push!(cdf, sum(seq .>= d) / length(seq))
            push!(k, d)
        end
        push!(cdfs, cdf)
        push!(ks, k)
    end
    return ks, cdfs
end

function powerlaw_plots(ks, cdfs, betas, seq_min, seq_max)
    plt = plot()
    for i in 1:3
        plot!(plt, ks[i], cdfs[i], xaxis=:log, yaxis=:log, color=i, linestyle=:dot, linewidth=2, label="γ=$(betas[i])")
        denom = sum(collect(seq_min:seq_max) .^ -betas[i])
        num = [sum(collect(k:seq_max) .^ -betas[i]) for k in seq_min:seq_max]
        plot!(plt, seq_min:seq_max, num ./ denom, xaxis=:log, yaxis=:log, color=i, linewidth=2, linestyle=:dash, label=nothing)
    end
    display(plt)
    return plt
end

function flatten_clustering(a::Vector{Vector{Int}})::Vector{Int}
    n = length.(a) |> sum
    flat_clustering = zeros(Int, n)
    for i in 1:length(a)
        com = a[i]
        for node in com
            flat_clustering[node] = i
        end
    end
    return flat_clustering
end

function ami_parameter_sweep(
    seq::Vector{<:Number},
    param_pos::Int,
    fixed_params,
    n::Int,
    d::Int,
    iter::Int,
    r::Float64=1.0,
    seed::Int=42)
    Random.seed!(seed)
    x = MLNABCDGraphGenerator.sample_points(n, d)
    amis = []
    for n1 in seq
        for n2 in seq
            println("Processing n₁: $(n1) and n₂: $(n2)")
            iter_amis = []
            params1 = deepcopy(fixed_params)
            params1[param_pos] = n1
            params2 = deepcopy(fixed_params)
            params2[param_pos] = n2
            for _ in 1:iter
                c1 = ABCDGraphGenerator.sample_communities(params1...)
                c2 = ABCDGraphGenerator.sample_communities(params2...)
                a1 = MLNABCDGraphGenerator.assign_points(x, c1)
                a2 = MLNABCDGraphGenerator.assign_points(x, c2)
                shuffled_flat_a1 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a1))
                shuffled_flat_a2 = flatten_clustering(MLNABCDGraphGenerator.shuffle_communities(r, a2))
                push!(iter_amis, AMI(shuffled_flat_a1, shuffled_flat_a2))
            end
            push!(amis, (mean(iter_amis), std(iter_amis), n1, n2, iter))
        end
    end
    return amis
end

function plot_cluster_stats(rs, stats_avgs, stats_stds, labels, dim)
    plt = plot(ylims=(0, 1))
    for i in 1:length(labels)
        plot!(plt, rs, stats_avgs[i], ribbon=stats_stds[i],
            markershape=:hexagon, markersize=2, markeralpha=0.5,
            fillalpha=0.5, label=labels[i], linewidth=2)
    end
    title!("Mean AMI $(dim)D")
    xlabel!("r")
    ylabel!("AMI")
    return plt
end

function assign_points_1d(x, c)
    @assert ndims(x) == 1
    @assert sum(c) == length(x)
    c = shuffle(c)
    x = copy(x)
    all_idxs = collect(1:length(x))
    dist = abs.(x)
    res = Vector{Int}[]
    for com in c
        ind = argmax(dist)
        ref = x[ind]
        dist_c = [abs(r - ref) for r in x]
        idxs = partialsortperm(dist_c, 1:com)
        push!(res, all_idxs[idxs])
        to_keep = setdiff(1:length(x), idxs)
        x = x[to_keep]
        dist = dist[to_keep]
        all_idxs = all_idxs[to_keep]
    end
    @assert length(x) == 0
    @assert length(all_idxs) == 0
    @assert length(dist) == 0
    @assert sort(union(res...)) == 1:sum(c)
    return res
end

function figsave(plt, plot_file)
    savefig(plt, plot_file)
    println("Saved $(plot_file)")
end