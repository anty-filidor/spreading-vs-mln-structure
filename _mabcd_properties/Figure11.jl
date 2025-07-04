# To obtain data for the plots run:
# julia --project experiments/edges_cor_convergence.jl
using DelimitedFiles
using Plots

graphs_path = "experiments/real-world-graphs/"
layers_limit = 5
graphs = ["ckm_physicians", "l2_course_net_1", "lazega", "timik1q2009"]
epsils = ["1", "5"]

graph_name_dict = Dict("ckm_physicians" => "ckmp",
    "l2_course_net_1" => "l2-course",
    "lazega" => "lazega",
    "timik1q2009" => "timik",
    "cannes" => "cannes"
)
for graph in graphs
    for epsil in epsils
        layers_params = readdlm("$(graphs_path)$(graph)/layer_params.csv")
        size(layers_params)[1] > layers_limit + 1 && continue
        ecor_log_file = filter(x -> contains(x, "00$(epsil).txt"), readdir("$(graphs_path)$(graph)", join=true))[1]
        ecor_log = readdlm(ecor_log_file, ',')
        mat_distances = convert.(Float64, ecor_log[:, end][2:end])
        ecor_log = ecor_log[:, 1:end-1]
        ecor_pairs = split.(ecor_log[1, :], '-')
        ecor_values = convert.(Float64, ecor_log[2:end, :])
        steps = size(ecor_values)[1]
        ecor_matrix = readdlm("$(graphs_path)$(graph)/edges_cor_matrix.csv", ',')
        ecor_matrix_vals = ecor_matrix[2:end, 2:end]
        layer_names = ecor_matrix[2:end, 1]
        plt = plot(ylims=(0, 0.48))
        yaxis2 = twinx()
        plot!(yaxis2, mat_distances, linewidth=2, alpha=0.8, color="gray", label=nothing)
        ylims!(yaxis2, (0, maximum(mat_distances) * 1.2))
        for (i, pair) in enumerate(ecor_pairs)
            l1, l2 = parse.(Int, pair)
            plot!(plt, ecor_values[:, i] .+ ecor_matrix_vals[l1, l2], color=i, label="$(layer_names[l1])-$(layer_names[l2])")
            plot!(plt, fill(ecor_matrix_vals[l1, l2], steps), color=i, linestyle=:dash, label=nothing)
        end
        if graph != "l2_course_net_1"
            plot!(legend=(0.69, 0.28))
        end
        if graph == "timik1q2009"
            plot!(legend=:topright)
        end
        title!("$(graph_name_dict[graph]), ϵ=0.0$(epsil)")
        xlabel!(plt, "Batch")
        xlabel!(yaxis2, "")
        ylabel!(plt, "Edges correlation")
        ylabel!(yaxis2, "L2 distance")
        plot_file = "img/edges_cor_convergence_$(graph)_$(epsil).pdf"
        savefig(plt, plot_file)
        println("Saved $(plot_file) for epsilon $(epsil)")
    end
end
