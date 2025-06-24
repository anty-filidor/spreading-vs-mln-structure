using Plots
using Random
using MLNABCDGraphGenerator
using ABCDGraphGenerator

include("utilities.jl")

###############################################
# Empirical vs Theoretical powerlaw - degrees #
###############################################

seed = 42
n = 100_000
betas = [2.2, 2.5, 2.8]
delta = 5
Delta = 316
max_iter = 1000

Random.seed!(seed)
degs_plot = [ABCDGraphGenerator.sample_degrees(beta, delta, Delta, n, max_iter) for beta in betas]
ks, cdfs = powerlaw_cdfs(degs_plot)
plt = powerlaw_plots(ks, cdfs, betas, delta, Delta)
xlabel!("degree")
ylabel!("1-cdf")
figsave(plt, "img/degrees_powerlaw.pdf")

#######################################################
# Empirical vs Theoretical powerlaw - community sizes #
#######################################################

seed = 42
n = 100_000
betas = [1.2, 1.5, 1.8]
s = 10
S = 1000
max_iter = 1000

Random.seed!(seed)
coms_plot = [ABCDGraphGenerator.sample_communities(beta, s, S, n, max_iter) for beta in betas]
ks, cdfs = powerlaw_cdfs(coms_plot)
plt = powerlaw_plots(ks, cdfs, betas, s, S)
xlabel!("community size")
ylabel!("1-cdf")
figsave(plt, "img/com_sizes_powerlaw.pdf")
