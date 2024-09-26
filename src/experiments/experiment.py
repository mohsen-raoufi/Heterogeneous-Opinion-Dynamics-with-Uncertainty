

from models.agent import BasicBayesianAgent, NaiveAgent, CIBayesianAgent, NaiveLocalOptimal, BayesCorrel, BayesCICorrel, NaiveCorrel, NaiveLOCorrel, BayesCICorrelMismodel, BayesCorrelMismodel, BayesCorrelCircularPlot
from models.collective import NetworkParams, Network
from models.environment import BasicGaussianEnvironment, GaussianEnvironmentExternalSTD, GaussianEnvironmentFixedSTD
from models.simulator import Simulator
import matplotlib.pyplot as plt
import datetime
from dataclasses import dataclass, field
import torch
import numpy as np
import random
from util import load_config
import argparse


@dataclass
class ExperimentParameters:
    run_name: str = "basic_experiment"
    time_str: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    seed: int = 0
    true_value: float = 0.75
    env_noise_std: float = 0.5
    shape: (int, int) = (1000, 1000)
    steps: int = 10
    visualize: bool = False
    environment_type: str = "Old"
    hack_for_circular_plot: bool = False

    agent_type: str = "Bayes"
    network_params: NetworkParams = field(default_factory=NetworkParams)


def run_experiment_from_files(file_path):
    config = load_config(ExperimentParameters, file_path+"/config")
    run_experiment(config, file_path)


def run_experiment(params, file_path):
    global collective, simulator
    # Setting the seed
    torch.autograd.set_detect_anomaly(True)
    torch.random.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    if params.environment_type == "Old":
        env = BasicGaussianEnvironment(params.true_value, params.env_noise_std, params.shape)
    elif params.environment_type == "GaussianFixedSTD":
        env = GaussianEnvironmentFixedSTD(params.true_value, params.env_noise_std)
    elif params.environment_type == "GaussianExternalSTD":
        env = GaussianEnvironmentExternalSTD(params.true_value)
    else:
        raise ValueError("unknown environment type: "+str(params.agent_type))


    if params.agent_type == "Bayes":
        agent_class = BasicBayesianAgent
    elif params.agent_type == "Naive":
        agent_class = NaiveAgent
    elif params.agent_type == "NaiveLO":
        agent_class = NaiveLocalOptimal
    elif params.agent_type == "BayesCI":
        agent_class = CIBayesianAgent
    elif params.agent_type == "BayesCORR":
        agent_class = BayesCorrel
    elif params.agent_type == "NaiveCORR":
        agent_class = NaiveCorrel
    elif params.agent_type == "NaiveLOCORR":
        agent_class = NaiveLOCorrel
    elif params.agent_type == "BayesCICORR":
        agent_class = BayesCICorrel
    elif params.agent_type == "BayesCORRMismodel":
        agent_class = BayesCorrelMismodel
    elif params.agent_type == "BayesCICORRMismodel":
        agent_class = BayesCICorrelMismodel
    elif params.agent_type == "BayesCircularPlot":
        agent_class = BayesCorrelCircularPlot
    else:
        raise ValueError("unknown agent type: "+str(params.agent_type))

    collective = Network(env, agent_class, params.network_params, params.seed, hack_for_circular_plot=params.hack_for_circular_plot)
    collective.hack_for_circular_plot = params.hack_for_circular_plot
    simulator = Simulator(env, collective)
    # setting a common color palette
    cmap = plt.get_cmap("plasma")
    env.set_belief_cmap(cmap)
    collective.set_belief_cmap(cmap)
    simulator.record_current_state(float(0))
    for i in range(params.steps):
        simulator.step()
        simulator.record_current_state(float(i+1))
        simulator.make_plots(None, params.visualize)
        if params.visualize:
            plt.pause(1)
    simulator.save_recorded_data(file_path + "/data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args, _ = parser.parse_known_args()
    run_experiment_from_files(args.filepath)

