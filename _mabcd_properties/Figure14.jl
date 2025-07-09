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
df_no_p6 = df[:, Not("edges_correlation_time")]
df_no_p6[!, "time"] = sum.(eachrow(df_no_p6[!, names(df_no_p6, r".time")]))

df[!, "time"] = sum.(eachrow(df[!, names(df, r".time")]))
for col in names(df, r"_time")
    df[!, "$(col)_perc"] = df[!, col] ./ df[!, "time"] .* 100
end
df_agg_perc = combine(groupby(df, [:n, :l]), names(df, r"_perc") .=> mean)
df_agg_perc = df_agg_perc[df_agg_perc.n.==65536, :]
plt = groupedbar(
    Matrix(df_agg_perc[!, Not([:n, :l])]),
    bar_position=:stack,
    bar_width=0.7,
    xticks=(1:length(df_agg_perc.l), df_agg_perc.l),
    ylims=(95, 100),
    label=["Phase 1" "Phase 2" "Phase 3" "Phase 4+5" "Phase 6"],
    tickfontsize=10,
    legendfontsize=10
)
xlabel!("l")
ylabel!("% of execution time")
figsave(plt, "img/execution_time_perc_with_p6.pdf")

for col in names(df_no_p6, r"_time")
    df_no_p6[!, "$(col)_perc"] = df_no_p6[!, col] ./ df_no_p6[!, "time"] .* 100
end
df_agg_perc_no_p6 = combine(groupby(df_no_p6, [:n, :l]), names(df_no_p6, r"_perc") .=> mean)
df_agg_perc_no_p6 = df_agg_perc_no_p6[df_agg_perc_no_p6.l.==5, :]

plt = areaplot(
    df_agg_perc_no_p6.n,
    -1 .* Matrix(df_agg_perc_no_p6[!, Not([:n, :l])]),
    xticks=([1024, 250000, 500000, 750000, 1000000],
        ["1024", "250 000", "500 000", "750 000", "1 000 000"]),
    yformatter=yi -> yi + 100,
    label=["Phase 1" "Phase 2" "Phase 3" "Phase 4+5"],
    tickfontsize=10,
    legendfontsize=10
)
xlabel!("n")
ylabel!("% of execution time")
figsave(plt, "img/execution_time_perc_without_p6_stacked.pdf")