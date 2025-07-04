# Spreading Effectiveness Versus Structure of the Network

A repository for evaluating the effectiveness of information diffusion in multilayer networks with
respect to their structural properties using mABCD. The project also contains a Julia module which
evaluates properties of the employed framework (i.e. mABCD).

**Authors**: Łukasz Kraiński (†), Michał Czuba (¶), Piotr Bródka (¶), Paweł Prałat (¬),
  Bogumił Kamiński (†), François Théberge (§)

- (¶) WUST, Wrocław, Lower Silesia, Poland
- (†) SGH, Warsaw, Masovia, Poland
- (¬) TMU, Toronto, Ontario, Canada
- (§) TIMC, Ottawa, Ontario, Canada

This repository is an complementary artifact for the paper: <TODO: add an URL>

## Evaluating Properties of mABCD

An additional functionality of the source code stored in this repository is an evaluation of
effectivenss of mABCD while generating graphs of different properties. It can be considered also as
a standalone project. For details see `_mabcd_properties` directory.

## Methodology

The methodology employed for assessing how the network topology affects an information diffusion
consists of following steps:
- analyse a real network;
- obtain a configuration model compatible with mABCD;
- modify the configuration to adjust the network structure;
- generate an artificial network based on the configuration;
- simulate a diffusion on both the real and modified graphs;
- compare the results.

### Series of Experiments

Baseline:
- 1: a twin of timik1q2009 with some parameters smoothed

Experiment A - modify the noise level between communities:
- 2: series 1 with modified xi in all layers to 1.00
- 3: series 1 with modified xi in all layers to 200% of the original xi
- 4: series 1 with modified xi in all layers to 50% of the original xi
- 5: series 1 with modified xi in all layers to 0.01

Experiment B - modify the number of actors:
- 6: series 1 with 150% of the original number of actors
- 7: series 1 with 125% of the original number of actors
- 8: series 1 with 75% of the original number of actors
- 9: series 1 with 50% of the original number of actors

Experiment C - modify the communities correlation between layers
- 10: series 1 with modified r to 1.000 (strenghten overlapping between comm-s)
- 11: series 1 with modified r to 0.667 (strenghten overlapping between comm-s)
- 12: series 1 with modified r to 0.333 (weaken overlapping between comm-s)
- 13: series 1 with modified r to 0.001 (weaken overlapping between comm-s)

## Structure of the Repository

```bash
.
├── README.md
├── _mabcd_properties        -> Experiments of mABCD properties, see README inside for details
├── data                     -> Use DVC to fetch this folder
│   ├── nets_generated       -> Generated networks with mABCD
│   ├── nets_properties      -> Properties of real networks used in experiments
│   ├── networks             -> Real networks used in experiments
│   ├── results_raw          -> Raw results from the conducted experiments
│   ├── results_processed    -> Processed results presented in the paper
│   └── test                 -> Data used in the E2E test
├── env
├── scripts                  -> Scripts for processing `data` with `src`
│   ├── analysis
│   └── configs
├── src                      -> Main code used by various scripts
│   ├── aux
│   ├── loaders
│   ├── mln_abcd             -> Python ports for mABCD
│   ├── simulator
│   ├── generator.py
│   ├── params_handler.py
│   ├── result_handler.py
│   └── utils.py
├── pyproject.toml
├── run_experiments.py       -> Main entry point for `src`
└── test_reproducibility.py  -> Simple E2E test to verify code reproducibility
```

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

The dataset is stored using DVC. To access it, a permission to access Google Drive is required.
Please send requests via email to michal.czuba@pwr.edu.pl in order to grant the access. Then,
execute the following command in the shell:

```bash
dvc pull
```

## Usage

To run the code execute: `python run_experiments.py <path to the configuration file>`. There are two
main functionalities of this codebase discussed below.

### Network Generator

The first major functionality of this repository is the generation of artificial multilayer networks
using MLNABCD (https://github.com/KrainskiL/MLNABCDGraphGenerator.jl), which provides a
fully-fledged Julia wrapper for Python. See `scripts/configs/example_generate.yaml` for reference.

### Diffusion Simulator

The second key functionality is the simulation of diffusion under the Multilayer Independent
Cascade Model. See `scripts/configs/example_simulate.yaml` for reference. For each repetition of
the Cartesian product computed for the provided parameters, a `csv` file will be generated with the
following columns:

```python
{
    seed_ids: str           # IDs of actors that were seeds, aggregated into a string (sep. by ;)
    gain: float             # Gain* obtained using this seed set
    area: float             # Area* under normalised activations curve obtained using this seed set
    simulation_length: int  # Number of simulation steps
    seed_nb: int            # Number of actors that were seeds
    exposed_nb: int         # Number of active actors at the end of the simulation
    unexposed_nb: int       # Number of actors that remained inactive
    expositions_rec: str    # Record of new activations per epoch, aggregated into a string (sep. ;)
    network_type: str       # Network's type
    network_name: str       # Network's name
    ss_method: str          # Seed selection method's name
    seed_budget: float      # Value of the maximum seed budget
    protocol: str           # Protocol's name
    probab: float           # Value of the activation probability
}
```

The simulator will also save the provided configuration file, and rankings of actors obtained with a
given seed selection method.

#### Results Reproducibility

Results are expected to be reproducible. This is verified by the test: `test_reproducibility.py`.

### Analysing Results

To process raw results, execute the scripts in the `scripts/analysis` directory in the order shown
in the following structure. The script names correspond to the names of the generated files
under `data`:

```bash
.
├── correlations_verbose.py  -> analyse a real network
├── degree_sequences.py      -> analyse a real network
├── configuration_model.py   -> obtain a configuration parameters compatible with mABCD
├── networks_eda.py          -> analyse networks generated with mABCD
└── process_results.py       -> analyse results of the experiment
```
