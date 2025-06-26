using MLNABCDGraphGenerator
using Random

iters = 10
nns = 2 .^ (10:17)
ls = [2, 3, 4, 5]
io = open("experiments_results/speed_experiment_results.csv", "a")
for n in nns
    for l in ls
        for iter in 1:iters
            println("n=$(n), l=$(l), iter: $(iter)")
            now = time()
            config = MLNABCDGraphGenerator.MLNConfig(1,
                n,
                "edges_cor.csv",
                "experiments/time_experiment_layer_params.csv",
                1000,
                1000,
                100,
                0.01,
                2,
                "edges.dat",
                "coms.dat",
                l,
                fill(0.5, l),
                round.(Int, n .* fill(0.5, l)),
                fill(0.5, l),
                fill(0.5, l),
                fill(2.5, l),
                fill(ceil(Int, 0.1 * sqrt(n)), l),
                fill(ceil(Int, sqrt(n)), l),
                fill(1.5, l),
                fill(ceil(Int, 0.005 * n), l),
                fill(ceil(Int, 0.15 * n), l),
                fill(0.5, l),
                false,
                fill(0.5, l, l)
            )
            Random.seed!(iter)
            #Active nodes
            active_nodes = MLNABCDGraphGenerator.generate_active_nodes(config)
            active_time = time() - now
            now = time()
            #Degree Sequences
            degrees = MLNABCDGraphGenerator.generate_degrees(config, active_nodes, false)
            degree_sampling_time = time() - now
            now = time()
            #Sizes of communities
            com_sizes, coms = MLNABCDGraphGenerator.generate_communities(config, active_nodes)
            coms_sampling_time = time() - now
            now = time()
            edges = MLNABCDGraphGenerator.generate_abcd(config, degrees, com_sizes, coms)
            edges_gen_time = time() - now
            now = time()
            edges = MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
            coms = MLNABCDGraphGenerator.map_communities_to_agents(n, coms, active_nodes)
            edges_rewired = MLNABCDGraphGenerator.adjust_edges_correlation(config, edges, coms, active_nodes, false, false)
            edges_correlation_time = time() - now
            data = "$(iter),$(n),$(l),$(active_time),$(degree_sampling_time),$(coms_sampling_time),$(edges_gen_time),$(edges_correlation_time)"
            println(data)
            write(io, data * "\n")
            flush(io)
        end
    end
end
close(io)