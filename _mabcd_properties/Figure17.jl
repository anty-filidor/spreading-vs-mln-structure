# To obtain data for the plots run:
# julia --project experiments/xi_timik.jl
using DataFrames
using DelimitedFiles
using CSV
using StatsBase
using Plots
using LaTeXStrings

include("utilities.jl")

graph = "timik1q2009"
suffix = "timik"
header = ["iter", "xi_input", "xi_l1", "xi_l2", "xi_l3", "r12", "r13", "r23"]
df = CSV.read("experiments_results/xi_experiments_results_$(suffix).csv", DataFrame, header=header)
for l in ["l1", "l2", "l3"]
    df[!, "xi_$(l)_ratio"] = df[!, "xi_$(l)"] ./ df[!, "xi_input"]
end
cor_mat = readdlm("experiments/real-world-graphs/$(graph)/edges_cor_matrix.csv", ',')
lnames = cor_mat[1, 2:end]
cor_mat = cor_mat[2:end, 2:end]
for cor in ["12", "13", "23"]
    df[!, "r$(cor)_ratio"] = df[!, "r$(cor)"] ./ cor_mat[parse.(Int, collect(cor))...]
end
df_agg = combine(groupby(df, :xi_input), names(df, r".ratio") .=> [mean std])
plt = plot(ylims=(0, 8.0), xticks=0.1:0.1:0.9, yticks=0.0:1.0:8.0)
for i in eachindex(lnames)
    plot!(plt, df_agg.xi_input, df_agg[!, "xi_l$(i)_ratio_mean"], ribbon=df_agg[!, "xi_l$(i)_ratio_std"],
        markershape=:hexagon, markersize=2, markeralpha=0.5, fillalpha=0.3, label="Layer $(i) - $(lnames[i])")
end
plot!(plt, df_agg.xi_input, fill(1.0, length(df_agg.xi_input)), linestyle=:dash, color="black", label=nothing)
display(plt)
xlabel!(L"ξ")
ylabel!(L"\hat{\xi}/ξ")
figsave(plt, "img/xi_ratio_$(suffix).pdf")

plt = plot(ylims=(0, 1.2), xticks=0.1:0.1:0.9)
for cor in ["12", "13", "23"]
    l1 = parse(Int, cor[1])
    l2 = parse(Int, cor[2])
    plot!(plt, df_agg.xi_input, df_agg[!, "r$(cor)_ratio_mean"], ribbon=df_agg[!, "r$(cor)_ratio_std"],
        markershape=:hexagon, markersize=2, markeralpha=0.5, fillalpha=0.3, label="$(l1)-$(l2) $(lnames[l1])-$(lnames[l2]) r=$(round(cor_mat[l1,l2],digits=2))")
end
plot!(plt, df_agg.xi_input, fill(1.0, length(df_agg.xi_input)), linestyle=:dash, color="black", label=nothing)
display(plt)
xlabel!(L"ξ")
ylabel!(L"\hat{r}/r")
figsave(plt, "img/r_ratio_$(suffix).pdf")