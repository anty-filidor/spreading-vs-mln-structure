# Spreading Effectiveness Versus Structure of the Network

A repository to check effectiveness of information diffusion in multilayer networks with regards to
their structural properties.

* Authors: Michał Czuba(¶†), ...
* Affiliation:  
        (¶) WUST, Wrocław, Lower Silesia, Poland  
        (†) ...

## Configuration of the runtime

First, initialise the enviornment:

```bash
conda env create -f env/conda.yaml
conda activate spreading-vs-mln-structure
python -m ipykernel install --user --name=spreading-vs-mln-structure
```

## Data

Dataset is stored on a DVC remote. Thus, to obtain it you have to access a Google Drive. Please
send a request via e-mail (michal.czuba@pwr.edu.pl) to have it granted. Then, simply execute in
the shell: `dvc pull`. **The dataset is large, hence we recommend to pull `zip` files only if
necessary.** For normal usage it is engouh to pull networks (`dvc pull data/networks`) and raw
results which are subjects of the analysis.


## Structure of the repository

```bash
.
├── README.md
├── data
│   ├── networks            -> networks used in experiments
│   ├── processed_results
│   ├── raw_results
│   └── test                -> examplary results of the simulator used in the E2E test
├── env                     -> a definition of the runtime environment
├── scripts
│   ├── analysis
│   └── configs             -> exemplary configuration files
├── src                     -> scripts to execute experiments and process the results
├── run_experiments.py      -> ...
├── test_reproducibility.py -> E2E test to prove that results can be repeated
```

## Running the pipeline

To run experiments execute: `python run_experiments.py <config file>`. See `example_config.yaml` for
inspirations. As a result, for each repetition of the cartesian product computed for the provided
parameters, a csv file will be obtained with following columns:

```python
{
    seed_ids: str           # IDs of actors that were seeds aggr. into string (sep. by ;)
    gain: float             # gain* obtained using this seed set
    simulation_length: int  # nb. of simulation steps
    seed_nb: int            # nb. of actors that were seeds
    exposed_nb: int         # nb. of active actors at the end of the simulation
    unexposed_nb: int       # nb. of actors that remained inactive
    expositons_rec: str     # record of new activations in each epoch aggr. into string (sep. by ;)
    network: str            # network's name
    protocol: str           # protocols's name
    seed_budget: float      # a value of the maximal seed budget
    mi_value: float         # a value of the threshold
    ss_method: str          # seed selection method's name
}
```

`*` Gain is the percentage of the non-initially seeded population that became exposed during the
simulation: `(exposed_nb - seed_nb) / (total_actor_nb - seed_nb) * 100%`

The simulator will also save provided configuraiton file, rankings of actors used in computations,
and detailed logs of evaluated cases whose index divided modulo by `full_output_frequency` equals 0.

## Results reproducibility

Results are supposed to be fully reproducable. There is a test for that: `test_reproducibility.py`.

## Obtaining analysis of results

To process raw results please execute scripts in `scripts/analysis` directory in the order as 
depicted in a following tree. Please note, that names of scripts reflect names of genreated files
under `data/processed_results`:

```bash
.
...
```
