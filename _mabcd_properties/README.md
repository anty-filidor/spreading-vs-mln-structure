## Environment Preparation

Install [Julia](https://julialang.org/install/). The code was tested using Julia v1.11.5
(2025-04-14). After successful installation run in Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Experiments Execution and Plots Generation

Create desired figure by running appropriate file, e.g. for Figure 3:

```julia
julia --project Figure3.jl
```
Some of the figures (12-15) rely on results produced with separate script. Please refer to the
comments in `.jl` files for details. For convenience results obtained during article preparation
are available in the `experiments_results` folder.
