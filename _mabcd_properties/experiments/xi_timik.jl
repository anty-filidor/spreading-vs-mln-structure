using DelimitedFiles
using MLNABCDGraphGenerator
using Random
using StatsBase

graph = "timik1q2009"
suffix = "timik"
seed = 42
iters = 25
xis = 0.1:0.1:0.9
graph_folder = "experiments/real-world-graphs/$(graph)/"
cwd = pwd()
cd(graph_folder)
config = MLNABCDGraphGenerator.parse_config("config.toml")
config.t = 300
config.eps = 0.01
l = config.l
deg_seq = readdlm("degrees.csv", ',')
deg_seq = deg_seq[sortperm(deg_seq[:, 1]), :][2:end, 2:end]
#Active nodes
active_nodes = [findall(x -> !isempty(x), row) for row in eachrow(deg_seq)]
#Degree Sequences
degrees = [trunc.(Ref(Int), filter(x -> !isempty(x), row)) for row in eachrow(deg_seq)]

cd(cwd)
io = open("experiments_results/xi_experiments_results_$(suffix).csv", "a")
for xi in xis
    for iter in 1:iters
        Random.seed!(seed+iter)
        now = time()
        println("ξ=$(xi), iter: $(iter)")
        config.xis = fill(xi, l)
        Random.seed!(iter)
        #Sizes of communities
        com_sizes, coms = MLNABCDGraphGenerator.generate_communities(config, active_nodes)
        #Generate ABCD graphs
        edges = MLNABCDGraphGenerator.generate_abcd(config, degrees, com_sizes, coms)
        #Map nodes and communities to agents
        edges = MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
        coms = MLNABCDGraphGenerator.map_communities_to_agents(config.n, coms, active_nodes)
        edges_rewired = MLNABCDGraphGenerator.adjust_edges_correlation(config, edges, coms, active_nodes, false, false)
        common_agents = MLNABCDGraphGenerator.common_agents_dict(active_nodes)
        edges_common_agents = MLNABCDGraphGenerator.common_agents_edges(edges_rewired, common_agents)
        edges_cor, cor_diff, cor_dist = MLNABCDGraphGenerator.calculate_edges_cor(edges_common_agents, config.edges_cor_matrix)
        gen_xis = [mean([coms[layer][e[1]] != coms[layer][e[2]] for e in edges_rewired[layer]]) for layer in 1:l]
        cors = ""
        for i in 1:l
            for j in (i+1):l
                r = edges_cor[i, j]
                cors *= "," * string(r)
            end
        end

        println("$(iter),$(xi),$(gen_xis[1]),$(gen_xis[2]),$(gen_xis[3])$(cors)")
        write(io, "$(iter),$(xi),$(gen_xis[1]),$(gen_xis[2]),$(gen_xis[3])$(cors)\n")
        flush(io)
        println(time() - now)
    end
end
close(io)