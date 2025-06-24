using DelimitedFiles
using MLNABCDGraphGenerator
using Random

graphs_folder = "experiments/real-world-graphs/"
graphs = ["ckm_physicians", "l2_course_net_1", "lazega", "timik1q2009"]
batch_params = [(0.05, 100), (0.01, 500)]
seed = 42

cwd = pwd()
for graph in graphs
    cd(graphs_folder * graph)
    for (eps, t) in batch_params
        Random.seed!(seed)
        now = time()
        config = MLNABCDGraphGenerator.parse_config("config.toml")
        config.eps = eps
        config.t = t
        deg_seq = readdlm("degrees.csv", ',')
        deg_seq = deg_seq[sortperm(deg_seq[:, 1]), :][2:end, 2:end]
        #Active nodes
        active_nodes = [findall(x -> !isempty(x), row) for row in eachrow(deg_seq)]
        #Degree Sequences
        degrees = [trunc.(Ref(Int), filter(x -> !isempty(x), row)) for row in eachrow(deg_seq)]
        #Sizes of communities
        com_sizes, coms = MLNABCDGraphGenerator.generate_communities(config, active_nodes)
        #Generate ABCD graphs
        edges = MLNABCDGraphGenerator.generate_abcd(config, degrees, com_sizes, coms)
        #Map nodes and communities to agents
        edges = MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
        coms = MLNABCDGraphGenerator.map_communities_to_agents(config.n, coms, active_nodes)
        #Adjust edges correlation
        edges_rewired = MLNABCDGraphGenerator.adjust_edges_correlation(config, edges, coms, active_nodes, false, true)
        str_eps = replace(string(eps), "." => "")
        log_file = filter(x -> endswith(x, ".log"), readdir())[1]
        mv(log_file, replace(log_file, ".log" => "_$(t)_$(str_eps).txt"))
        println(time() - now)
    end
    cd(cwd)
end