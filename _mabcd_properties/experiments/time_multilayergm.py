import multilayerGM as gm
from time import perf_counter
from math import sqrt, ceil

iters = 10
ls = [2, 3, 4, 5]
ns = [2**k for k in range(10, 17)]  # [1024, 2048, 4096, 8192, 16384, 32768, 65536]
io = open("experiments_results/speed_experiment_results_multilayergm.csv", "a")
io.write("iter,n,l,elapsed\n")
for n in ns:
    for l in ls:
        for i in range(iters):
            nowt = perf_counter()
            k_min = ceil(0.1 * sqrt(n))
            k_max = ceil(sqrt(n))
            n_sets = ceil(0.2 * sqrt(n))
            dt = gm.dependency_tensors.UniformMultiplex(n, l, 0.5)
            null = gm.DirichletNull(layers=dt.shape[1:], theta=1, n_sets=n_sets)
            partition = gm.sample_partition(
                dependency_tensor=dt, null_distribution=null
            )
            multinet = gm.multilayer_DCSBM_network(
                partition, mu=0.5, k_min=k_min, k_max=k_max, t_k=-2.5
            )
            elapsed = perf_counter() - nowt
            print("Iteration:", i, " Layers:", l, " n:", n, "Elapsed:", elapsed)
            io.write(f"{i},{n},{l},{elapsed}\n")
            io.flush()
io.close()
