# Spreading Effectiveness Versus Structure of the Network

A repository for evaluating the effectiveness of information diffusion in multilayer networks with
respect to their structural properties.

* **Authors**: Michał Czuba (¶†), ...
* **Affiliation**:  
  (¶) WUST, Wrocław, Lower Silesia, Poland  
  (†) ...

## Methodology

The methodology for assessing how network topology affects information diffusion is as follows:
- Analyse a real network (`scripts/correlations_verbose.py`).
- Obtain a configuration model compatible with MLNABCD (`scripts/configuration_model.py`).
- Modify the configuration to adjust the network structure.
- Generate an artificial network based on the configuration (`run_experiments.py`, `generate` mode).
- Simulate diffusion on both the real and modified networks (`run_experiments.py`, `simulation` mode).
- Compare the results (TBD).

## Runtime Configuration

First, initialise the environment:

```bash
conda env create -f env/conda.yaml
conda activate spreading-vs-mln-structure
python -m ipykernel install --user --name=spreading-vs-mln-structure
```

To use scripts that perform analysis, install the source code:

```bash
pip install -e .
```

## Data

The dataset is stored on a DVC remote. To access it, you need permission for Google Drive. Please
send a request via email (michal.czuba@pwr.edu.pl). Once granted, execute the following command in
the shell:

```bash
dvc pull
```

Series of experiments:
- 0a: timik1q2009
- 0b: artifitial networks generated on obtained configuration model for timik1q2009
- 1: 75% of actors from series 0
- 2: 50% of actors from series 0
- 3: 25% of actors from series 0
- 4: 1% of actors from series 0
- 5: xi in all layers equals to 0.01 (the original value was 0.001)
- 6: xi in all layers equals to 0.05
- 7: xi in all layers equals to 0.10
- 8: xi in all layers equals to 0.50
- 9: xi in all layers equals to 1.00
- 10: Delta equals to 1000 nodes per layer

## Repository Structure

```bash
.
├── README.md
├── data                     -> Use DVC to fetch this folder
│   ├── nets_properties      -> Properties of real networks used in experiments
│   ├── networks             -> Real networks used in experiments
│   └── test                 -> Data used in the E2E test
├── env
├── scripts                  -> Scripts for processing `data` with `src`
│   ├── analysis
│   └── configs
├── src                      -> Main code used by various scripts
│   ├── loaders
│   ├── mln_abcd
│   ├── simulator
│   └── generator.py
├── pyproject.toml
├── run_experiments.py       -> Main entry point for `src`
└── test_reproducibility.py  -> Simple E2E test to verify code reproducibility
```

## Network Generator

The first major functionality of this repository is the generation of artificial multilayer networks
using MLNABCD (https://github.com/KrainskiL/MLNABCDGraphGenerator.jl), which provides a fully-fledged
Julia wrapper for Python. See `scripts/configs/example_generate.yaml` for reference.

## Diffusion Simulator

The second key functionality of this repository is the simulation of diffusion under the Multilayer
Linear Threshold Model. See `example_config.yaml` for reference. For each repetition of the
Cartesian product computed for the provided parameters, a `csv` file will be generated with the
following columns:

```python
{
    seed_ids: str           # IDs of actors that were seeds, aggregated into a string (separated by ;)
    gain: float             # Gain* obtained using this seed set
    simulation_length: int  # Number of simulation steps
    seed_nb: int            # Number of actors that were seeds
    exposed_nb: int         # Number of active actors at the end of the simulation
    unexposed_nb: int       # Number of actors that remained inactive
    expositions_rec: str    # Record of new activations per epoch, aggregated into a string (sep. ;)
    network: str            # Network's name
    protocol: str           # Protocol's name
    seed_budget: float      # Value of the maximum seed budget
    mi_value: float         # Value of the threshold
    ss_method: str          # Seed selection method's name
}
```

`*` Gain represents the percentage of the non-initially seeded population that became exposed during
the simulation: `(exposed_nb - seed_nb) / (total_actor_nb - seed_nb) * 100%`

The simulator will also save the provided configuration file, rankings of actors used in
computations, and detailed logs of evaluated cases whose index modulo `full_output_frequency`equals 0.

### Results Reproducibility

Results are expected to be fully reproducible. This is verified by the test: `test_reproducibility.py`.

## Analysing Results

To process raw results, execute the scripts in the `scripts/analysis` directory in the order shown in
the following structure. The script names correspond to the names of the generated files under `data`:

```bash
.
├── degree_sequences.py
├── correlations_verbose.py
└── configuration_model.py
...
