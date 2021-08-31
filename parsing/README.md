# Dependency Parsing Experiments

This folder contains the necessary configuration files for running the dependency parsing experiments.

## MaChAmp Installation

We are using the MaChAmp toolkit (v0.2), a freely available AllenNLP-based toolkit available at: https://github.com/machamp-nlp/machamp .

To install it, clone the repository and install the Python requirements (ideally within a virtual environment):

```bash
(venv) $ pip3 install -r requirements.txt
```

## Running an Experiment

To run an experiment, please create the appropriate UD data subset according to the instructions in the top-level `README.md`. Point the dataset configuration files in `dataset-configs/` to the directory of the subset and train the parsers using the parameter configuration in `parameter-configs/` (only randoms seeds differ between parameter configurations).

```bash
(venv) $ python train.py --dataset_config data/subset.json --parameters_config params-rs42.json --device 0
```

## Provided Configurations

We provide the parameter and dataset configurations for our dependency parsing experiments in this directory.

### Parameter Configurations

Parameters follow MaChAmp's default configuration, but have been explicitly stated in order to ensure replicability. They can be found under `parameter-configs/params-rs*.json` with `rs-42` for example denoting the use of the random seed 42. All other parameters are the same across files.

### Dataset Configurations

Each dataset configuration points to the training and development splits of one of the UD subsets. Please change the absolute path to the one on your machine (removed for anonymization).

### Predictions

We provide the instance-level predictions of each parser and each random initialization as CoNLL-U files in [this archive](https://personads.me/x/emnlp-2021-data) in order ensure that future work can evaluate the statistical significance of performance differences. 