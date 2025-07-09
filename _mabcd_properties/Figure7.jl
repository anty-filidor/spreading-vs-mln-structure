using Plots
using PyCall
AMI = pyimport("sklearn.metrics").adjusted_mutual_info_score

include("utilities.jl")

################
# β₁ vs β₂ AMI #
################
n = 1000
iter = 500
betas = 1.05:0.05:2
c_min = 8
c_max = 32
max_iter = 1000
r = 1.0
seed = 42
d = 2

fixed = Number[0.0, c_min, c_max, n, max_iter]
amis = ami_parameter_sweep(collect(betas), 1, fixed, n, d, iter, r, seed)
reshaped_amis = reshape(getindex.(amis, 1), 20, 20)
plt = heatmap(betas, betas, reshaped_amis, ticks=betas[1:2:end])
title!("Mean AMI 2D")
xlabel!("β₁")
ylabel!("β₂")
figsave(plt, "img/ami_betas_heatmap_2D.pdf")

d = 1
amis_1d = ami_parameter_sweep(collect(betas), 1, fixed, n, d, iter, r, seed)
reshaped_amis_1d = reshape(getindex.(amis_1d, 1), 20, 20)
plt = heatmap(betas, betas, reshaped_amis_1d, ticks=betas[1:2:end])
title!("Mean AMI 1D")
xlabel!("β₁")
ylabel!("β₂")
figsave(plt, "img/ami_betas_heatmap_1D.pdf")
