<embed src="figures/Fig1.pdf" width="1000" type="application/pdf"/>

# Leveraging Uncertainty in Collective Opinion Dynamics with Heterogeneity

Code for the experiments and plots in our paper TODO ADD LINK

## Repository Structure

The code base consists of four main folders 
- models: containing code to build all Bayes and Naive variations in the experiments
- experiments: containing code to run experiments on these models
- plotting_and_analysis: contains code to analyse and plot executed experiments
- hpc_scripts: contains scripts to run experiments and do some prelimenary analysis on HPC clusters

## Setup

To setup generate a conda or venv with the env.yml file.

## Run an Experiment

To simply run an experiment try out src/experiments/local_test.py
There you can also easily play around with some parameters of the models or scenario.

## Run lots of Experiments (on HPC)

To run los of experiments try out src/experiments/experiment_runner.py or to run this on an HPC cluster src/hpc_scripts/hpc_slurm.bash

## Data & Plotting
Plotting tools can be found in plotting_and_analysis. To analyse our original data, please find the data under TODO ADD LINK
