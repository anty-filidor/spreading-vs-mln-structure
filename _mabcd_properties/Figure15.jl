# Please install mulitlayerGM-py:
# pip install git+https://github.com/MultilayerGM/MultilayerGM-py.git@master
# For more details visit https://github.com/MultilayerGM/MultilayerGM-py
# To obtain data for the plots run:
# julia --project experiments/time.jl
# julia --project experiments/time_no_phase6.jl
# python experiments/time_multilayergm.py
# Please note that the results are not deterministic as they rely on the hardware used
# Files `experiments_results/speed_experiment_results.csv` and `experiments_results/speed_experiment_results_multilayergm.csv` were generated on Macbook Air M1

using DataFrames
using DelimitedFiles
using CSV
using StatsBase
using Plots

include("utilities.jl"
)
df = CSV.read("experiments_results/speed_experiment_results.csv", DataFrame)
df_gm = CSV.read("experiments_results/speed_experiment_results_multilayergm.csv", DataFrame)

df = df[df.n.<40000, :]
df = df[df.edges_correlation_time.!=0.0, :]
df[!, "time"] = sum.(eachrow(df[!, names(df, r".time")]))
df_agg = combine(groupby(df, [:n, :l]), :time .=> [mean std])
df_agg_gm = combine(groupby(df_gm, [:n, :l]), :elapsed .=> [mean std])
plt = plot(legend=:bottomright, ylims=(0.01, 10000), xlims=(500, 70000))
for (i, l) in enumerate(unique(df_agg.l))
    df_tmp = df_agg[df_agg.l.==l, :]
    plot!(plt, df_tmp.n, df_tmp.time_mean, color=i, yaxis=:log, xaxis=:log, yerr=df_tmp.time_std, label="$(l) layers")
    df_tmp = df_agg_gm[df_agg_gm.l.==l, :]
    plot!(plt, df_tmp.n, df_tmp.elapsed_mean, color=i, linestyle=:dash, yaxis=:log, xaxis=:log, yerr=df_tmp.elapsed_std, label=" ")
end
xticks!(2 .^ (10:15), string.(2 .^ (10:15)))
display(plt)
xlabel!("n")
ylabel!("time (seconds)")
figsave(plt, "img/execution_time_log_multilayergm_comparison.pdf")

df_merged = innerjoin(df_agg, df_agg_gm, on=[:n, :l])
df_merged[!, "ratio"] = round.(Int64, df_merged.elapsed_mean ./ df_merged.time_mean)
df_gm_res = unstack(df_merged, :n, :l, :elapsed_mean, combine=x -> round.(x, digits=2)[1])
df_res = unstack(df_merged, :n, :l, :time_mean, combine=x -> round.(x, digits=2)[1])
df_ratio_res = unstack(df_merged, :n, :l, :ratio, combine=sum)