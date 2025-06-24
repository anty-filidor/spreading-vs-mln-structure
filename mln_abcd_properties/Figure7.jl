using Plots
using PyCall
AMI = pyimport("sklearn.metrics").adjusted_mutual_info_score

include("utilities.jl")

################
# S₁ vs S₂ AMI #
################
n = 1000
iter = 500
c_maxes = 16:16:320
c_min = 8
beta = 1.5
max_iter = 1000
r = 1.0
seed = 42
d = 2

fixed = Number[beta, c_min, 0, n, max_iter]
amis_cmax = ami_parameter_sweep(collect(c_maxes), 3, fixed, n, d, iter, r, seed)
reshaped_amis_cmax = reshape(getindex.(amis_cmax, 1), 20, 20)
plt = heatmap(c_maxes, c_maxes, reshaped_amis_cmax, ticks=c_maxes[1:2:end])
title!("Mean AMI 2D")
xlabel!("S₁")
ylabel!("S₂")
figsave(plt, "img/ami_S_heatmap_2D.pdf")

d = 1
amis_1d_cmax = ami_parameter_sweep(collect(c_maxes), 3, fixed, n, d, iter, r, seed)
reshaped_amis_1d_cmax = reshape(getindex.(amis_1d_cmax, 1), 20, 20)
plt = heatmap(c_maxes, c_maxes, reshaped_amis_1d_cmax, ticks=c_maxes[1:2:end])
title!("Mean AMI 1D")
xlabel!("S₁")
ylabel!("S₂")
figsave(plt, "img/ami_S_heatmap_1D.pdf")