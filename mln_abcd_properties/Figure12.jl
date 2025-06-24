# To obtain data for the plots run:
# julia --project experiments/time.jl
# julia --project experiments/time_no_phase6.jl
# Please note that the results are not deterministic as they rely on the hardware used
# File `experiments_results/speed_experiment_results.csv` was generated on Macbook Air M1

using DataFrames
using DelimitedFiles
using CSV
using StatsBase
using Plots
using StatsPlots

include("utilities.jl")

df = CSV.read("experiments_results/speed_experiment_results.csv", DataFrame)

df_p6 = df[df.n.<200000, :]
df_p6[!, "time"] = sum.(eachrow(df_p6[!, names(df_p6, r".time")]))
df_agg_p6 = combine(groupby(df_p6, [:n, :l]), :time .=> [mean std])
df_no_p6 = df[:, Not("edges_correlation_time")]
df_no_p6[!, "time"] = sum.(eachrow(df_no_p6[!, names(df_no_p6, r".time")]))
df_agg_no_p6 = combine(groupby(df_no_p6, [:n, :l]), :time .=> [mean std])
plt = plot(legend=:bottomright, ylims=(0.001, 1000), xlims=(500, 1500000), xticks=10 .^ (3:6), tickfontsize=10, legendfontsize=10)
for (i, l) in enumerate(unique(df_agg_no_p6.l))
    df_tmp = df_agg_no_p6[df_agg_no_p6.l.==l, :]
    plot!(plt, df_tmp.n, df_tmp.time_mean, color=i, yaxis=:log, xaxis=:log, yerr=df_tmp.time_std, label="$(l) layers")
    df_tmp = df_agg_p6[df_agg_p6.l.==l, :]
    plot!(plt, df_tmp.n, df_tmp.time_mean, color=i, linestyle=:dash, yaxis=:log, xaxis=:log, yerr=df_tmp.time_std, label=" ")
end
display(plt)
xlabel!("n")
ylabel!("time (seconds)")
figsave(plt, "img/execution_time_log.pdf")