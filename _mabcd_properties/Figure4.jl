using Plots
using Random
using MLNABCDGraphGenerator
using StatsBase

include("utilities.jl")

#########################
# Correlation tau-sigma #
#########################
seed = 42
xs = 0.01:0.1:20
ns = [1000, 10000, 100_000, 1_000_000]

ys_n = []
Random.seed!(seed)
for n in ns
    println("Sampling n=$(n)")
    vs = collect(1:n)
    ys = [corkendall(vs, MLNABCDGraphGenerator.sample_ranking(vs, n, x)) for x in xs]
    push!(ys_n, (n, ys))
end

plt = plot()
for (n, ys) in ys_n
    plot!(plt, xs, ys, ylim=[0, 1], label="n=$(n)", linewidth=2)
end
xlabel!("σ")
ylabel!("Kendall τ correlation")
figsave(plt, "img/degree_correlation_sigma.pdf")

plt = plot()
for i in 1:3
    scatter!(plt, xs, ys_n[end][2] - ys_n[i][2], label="n=$(ys_n[i][1])", markersize=4, markershape=:hexagon)
end
xlabel!("σ")
ylabel!("Kendall τ correlation difference")
figsave(plt, "img/degree_correlation_diff_sigma_dots.pdf")